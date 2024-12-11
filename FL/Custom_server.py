import os
import random
import torch
import time
import timeit
import flwr as fl
import numpy as np
from logging import INFO
import matplotlib.pyplot as plt
from flwr.common.logger import log
from collections import OrderedDict
from flwr.server.history import History
from flwr.server.server import fit_clients
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import Tuple, Optional, Dict, Union, List
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Code, FitRes, Parameters, GetParametersIns

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
from loss import combined_loss
from Custom_Strategy import CustomStrategy
from params_utils import gen_final_parameters
from Models import UNet_MRI, UNet_CT, UNet_US


class CustomServer(fl.server.Server):

    def __init__(
        self,
        strategy: CustomStrategy,
        client_manager: ClientManager,
        model_reduced_server,
        batch_size: int,
        testloader_server,
        trainloader_server_red,
        testloader_server_red,
        DEVICE: str,
        know_distillation: bool,
        modalities_clients: list[str],
        save_plots: bool,
    ):

        super().__init__(client_manager=client_manager, strategy=strategy)

        self.model_reduced_server = (
            model_reduced_server
        )
        self.testloader_server = (
            testloader_server
        )
        self.trainloader_server_red = (
            trainloader_server_red
        )
        self.testloader_server_red = (
            testloader_server_red
        )
        self.DEVICE = DEVICE
        self.know_distillation = know_distillation
        self.modalities_clients = modalities_clients
        self.batch_size = 2
        self.save_plots = save_plots
        self.dist_losses = []
        self.accuracy_list = []
        self.dice_list = []
        self.start_date = time.strftime("%Y%m%d_%H%M%S")
        self.KD_epochs = 1000
        self.lr = 1e-4
        self.train_dice_scores = []
        self.test_dice_scores = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_dice_epoch = []
        self.test_acc_epoch = []
        self.epoch_dist_losses = []
        self.modalities = sorted(list(set(modalities_clients)))

    def _get_initial_parameters(
        self, server_round: int, timeout: Optional[float]
    ) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial global parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=timeout, group_id=server_round
        )
        assert (
            get_parameters_res.status.code == Code.OK
        ), "Failed to receive initial parameters from the client"
        parameters = get_parameters_res.parameters

        parameters_array = parameters_to_ndarrays(parameters)

        aggregated_weights_by_modality = {}

        for client_id in range(len(self.modalities_clients)):
            modality = self.modalities_clients[client_id]
            if modality not in aggregated_weights_by_modality:
                aggregated_weights_by_modality[modality] = {}
            aggregated_weights_by_modality[modality][client_id] = [
                weight.copy() for weight in parameters_array
            ]

        final_parameters_array = gen_final_parameters(aggregated_weights_by_modality, 0)
        final_parameters = ndarrays_to_parameters(final_parameters_array)

        return final_parameters

    def _get_client_properties(self):
        """Get properties from all clients."""
        dict_properties = self.strategy.client_properties(
            client_manager=self._client_manager
        )
        print("Client properties:", dict_properties)
        return dict_properties

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        history = History()
        print("FIT server device:", self.DEVICE)
        self.model_reduced_server.to(self.DEVICE)

        # Initialize parameters
        log(INFO, "[INIT]")
        log(INFO, self.start_date)
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Evaluating initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Get client properties
        client_properties = self._get_client_properties()

        clients_per_modality = {}

        for cid, properties in client_properties.items():
            modality = properties["Modality"]
            client_id = properties["Client_ID"]
            if modality not in clients_per_modality:
                clients_per_modality[modality] = []

            clients_per_modality[modality].append((cid, client_id))

        clients_per_modality = {
            k: clients_per_modality[k] for k in sorted(clients_per_modality.keys())
        }

        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
                clients_per_modality=clients_per_modality,
            )
            weigths_server = None
            if res_fit is not None:

                (
                    parameters_aggregated,
                    metrics_aggregated,
                    weigths_server,
                    (results, failures),
                ) = res_fit
                if parameters_aggregated:
                    self.parameters = parameters_aggregated
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=metrics_aggregated
                )

            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

            # KNOWLEDGE DISTILLATION
            if self.know_distillation:
                if (
                    current_round % num_rounds == 0
                ):
                    if weigths_server is not None:

                        log(INFO, "  ")
                        log(INFO, " ---- KNOWLEDGE DISTILLATION ---- ")

                        # Extract logits from teacher models
                        teacher_models = {
                            "MRI": UNet_MRI(),
                            "CT": UNet_CT(),
                            "US": UNet_US(),
                        }
                        for k in teacher_models.keys():
                            if k in self.modalities:
                                params_dict = zip(
                                    teacher_models[k].state_dict().keys(),
                                    weigths_server[k],
                                )
                                state_dict = OrderedDict(
                                    {k: torch.tensor(v) for k, v in params_dict}
                                )
                                teacher_models[k].load_state_dict(
                                    state_dict, strict=True
                                )
                            else:
                                teacher_models.pop(k)

                        teacher_models = [model for model in teacher_models.values()]

                        for t in teacher_models:
                            print("teachers:", t.name)
                        teacher_models = [
                            model.to(self.DEVICE) for model in teacher_models
                        ]

                        teacher_logits_list = []
                        for inputs, _ in self.testloader_server:
                            inputs = inputs.float().to(self.DEVICE)

                            if inputs.size(0) < self.batch_size:
                                print(f"Skipping batch with size: {inputs.size(0)}")
                                continue

                            for model in teacher_models:
                                logits = model(
                                    inputs
                                ).detach()
                                teacher_logits_list.append(logits)
                        best_student_model_weights = (
                            self.model_reduced_server.state_dict()
                        )
                        for ep in range(self.KD_epochs):
                            print("KD epoch:", ep)
                            # Data augmentation
                            round_random = random.randint(0, 100 - 1)
                            epoch_random = random.randint(0, 10 - 1)

                            self.trainloader_server_red.dataset.set_round(
                                round_random
                            )
                            self.trainloader_server_red.dataset.set_epoch(
                                epoch_random
                            )
                            train_dice_epoch, train_acc_epoch, dist_losses_epoch = (
                                [],
                                [],
                                [],
                            )
                            optimizer = torch.optim.Adam(
                                self.model_reduced_server.parameters(), lr=self.lr
                            )
                            for (
                                inputs,
                                targets,
                            ) in (
                                self.trainloader_server_red
                            ): 
                                inputs = inputs.float().to(self.DEVICE)
                                targets = targets.float().to(self.DEVICE)
                               
                                distill_loss = self.knowledge_distillation(
                                    teacher_logits_list=teacher_logits_list,
                                    student_model=self.model_reduced_server,
                                    inputs=inputs,
                                    targets=targets,
                                    temperature=1.0,
                                )

                                # Perform backpropagation for the student model
                                optimizer.zero_grad()
                                distill_loss.backward()
                                optimizer.step()

                                loss_value = (
                                    distill_loss.item()
                                ) 
                                dist_losses_epoch.append(loss_value)

                                with torch.no_grad():
                                    student_logits = self.model_reduced_server(
                                        inputs
                                    )
                                    preds = (
                                        student_logits > 0.5
                                    ).float()
                                    accuracy = (preds == targets).float().mean().item()
                                    dice = self.dice_coefficient(preds, targets)
                                    train_acc_epoch.append(accuracy)
                                    train_dice_epoch.append(dice)

                            mean_dist_loss = sum(dist_losses_epoch) / len(
                                dist_losses_epoch
                            )
                            mean_train_dice = sum(train_dice_epoch) / len(
                                train_dice_epoch
                            )
                            mean_train_acc = sum(train_acc_epoch) / len(train_acc_epoch)
                            self.epoch_dist_losses.append(mean_dist_loss)
                            self.train_dice_scores.append(mean_train_dice)
                            self.train_accuracies.append(mean_train_acc)
                            best_student_model_weights = (
                                self.model_reduced_server.state_dict()
                            )
                            torch.save(
                                best_student_model_weights, "FL_KD_best_model.pth"
                            )

                        predicted_masks = []
                        dice_scores = []

                        with torch.no_grad():
                            for (
                                inputs_test,
                                true_masks,
                            ) in (
                                self.testloader_server_red
                            ): 
                                inputs_test = inputs_test.float().to(self.DEVICE)
                                true_masks = true_masks.float().to(self.DEVICE)
                                outputs = self.model_reduced_server(
                                    inputs_test
                                )
                                preds = (
                                    outputs > 0.5
                                ).float()
                                predicted_masks.append(
                                    preds.cpu()
                                )
                                self.test_dice_epoch.append(
                                    self.dice_coefficient(preds, true_masks)
                                )
                                self.test_acc_epoch.append(
                                    (preds == true_masks).float().mean().item()
                                )

                                dice = self.dice_coefficient(preds, true_masks)
                                dice_scores.append(dice)
                        mean_test_dice = sum(self.test_dice_epoch) / len(
                            self.test_dice_epoch
                        )
                        mean_test_acc = sum(self.test_acc_epoch) / len(
                            self.test_acc_epoch
                        )

                        self.test_dice_scores.append(mean_test_dice)
                        self.test_accuracies.append(mean_test_acc)

                        print(
                            f"Epoch {ep}/{self.KD_epochs}:"
                            f" Distillation Loss: {mean_dist_loss:.4f}"
                        )
                        print(
                            f"Epoch {ep}/{self.KD_epochs}:"
                            f" Train Dice: {mean_train_dice:.4f}, Test Dice: {mean_test_dice:.4f}"
                        )
                        print(
                            f"Epoch {ep}/{self.KD_epochs}:"
                            f" Train Accuracy: {mean_train_acc:.4f}, Test Accuracy: {mean_test_acc:.4f}"
                        )

                        predicted_masks = torch.cat(
                            predicted_masks
                        ).numpy()

                        self.plot_images_and_masks(
                            inputs_test.cpu().numpy(),
                            true_masks.cpu().numpy(),
                            predicted_masks,
                        )
                        self.plot_metrics(
                            self.epoch_dist_losses,
                            self.train_dice_scores,
                            self.test_dice_scores,
                            self.train_accuracies,
                            self.test_accuracies,
                        )
            # END KNOWLEDGE DISTILLATION

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
        clients_per_modality: Optional[Dict[str, List[str]]],
    ) -> Optional[
        Tuple[
            Tuple[Optional[Parameters], Dict[str, List[int]]],
            Dict[str, Scalar],
            FitResultsAndFailures,
        ]
    ]:
        """Perform a single round of federated averaging."""
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
            Dict[str, List],
        ] = self.strategy.aggregate_fit(
            server_round, results, failures, clients_per_modality
        )

        parameters_aggregated, metrics_aggregated, weigths_server = aggregated_result

        if not parameters_aggregated:
            log(INFO, "aggregate_fit: no parameters aggregated, cancel")
            return None

        return (
            parameters_aggregated,
            metrics_aggregated,
            weigths_server,
            (results, failures),
        )

    def knowledge_distillation(
        self, teacher_logits_list, student_model, inputs, targets, temperature=3.0
    ):
        """Perform knowledge distillation between teacher and student
        Args:
            teacher_logits_list: List of logits from teacher models
            student_model: Student model
            inputs: Input data
            targets: Target data
            temperature: Temperature for distillation
        """
        if not teacher_logits_list:
            print("No teacher logits available, skipping distillation.")
            return torch.tensor(
                0.0, device=self.DEVICE, requires_grad=True
            )

        student_model = student_model.to(self.DEVICE)
        print("temperature:", temperature)
        with torch.no_grad():
            teacher_logits_agg = torch.mean(torch.cat(teacher_logits_list), dim=0)

        student_logits = student_model(inputs)  # forward pass

        teacher_logits_agg = teacher_logits_agg.repeat(student_logits.size(0), 1, 1, 1)

        total_loss = combined_loss(
            pred=student_logits,
            target=targets,
            student_logits=student_logits,
            teacher_logits=teacher_logits_agg,
            distill_weight=0.7,
            T=temperature,
        )

        return total_loss

    def dice_coefficient(self, predicted_masks, masks, epsilon=1e-6):
        predicted_masks = (predicted_masks > 0.5).to(torch.bool)
        masks = (masks > 0.5).to(torch.bool)

        intersection = (predicted_masks & masks).float().sum((1, 2, 3))
        denominator = predicted_masks.float().sum((1, 2, 3)) + masks.float().sum(
            (1, 2, 3)
        )

        dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
        return dice.mean().item()

    def plot_images_and_masks(
        self, images, true_masks, predicted_masks, num_images=None
    ):
        print("Plotting knowledge distillation images...")
        if num_images is None:
            num_images = min(len(images), len(true_masks), len(predicted_masks))

        if images.ndim == 1:
            images = images.reshape(
                -1, 1, 28, 28
            )
        fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        if num_images == 1:
            axs = axs[np.newaxis, :]

        for i in range(num_images):
            image = images[i]
            true_mask = true_masks[i]
            predicted_mask = predicted_masks[i]

            axs[i, 0].imshow(image.squeeze(), cmap="gray")
            axs[i, 0].set_title("Real Image")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(true_mask.squeeze(), cmap="gray")
            axs[i, 1].set_title("True Mask")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(predicted_mask.squeeze(), cmap="gray")
            axs[i, 2].set_title("Predicted Mask")
            axs[i, 2].axis("off")

        plt.tight_layout()
        if self.save_plots:

            if not os.path.exists("Plots"):
                os.makedirs("Plots")

            plt.savefig(f"Plots/KDplot_{self.start_date}.png")
            plt.close(fig)
        else:
            plt.show()

    def plot_distill_loss(self, dist_losses):
        """Plot the distillation loss."""
        plt.figure(figsize=(10, 5))
        plt.plot(dist_losses, label="Distillation Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Knowledge Distillation Loss")
        plt.legend()
        plt.grid(True)
        if self.save_plots:
            if not os.path.exists("Plots"):
                os.makedirs("Plots")
            plt.savefig(f"Plots/DistillationLoss_{self.start_date}.png")
            plt.close()
        else:
            plt.show()

    def plot_dice_accuracy(self, dice_scores, accuracies):
        """
        Plotta i grafici della distillation loss, Dice coefficient e Accuracy in un unico grafico con subplots.
        """
        if not (len(dice_scores) == len(accuracies)):
            raise ValueError(
                "Le liste dice_scores e accuracies devono avere la stessa lunghezza."
            )

        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle("Knowledge Distillation Metrics", fontsize=16)

        # Dice Coefficient
        axs[0].plot(dice_scores, label="Dice Coefficient", color="green")
        axs[0].set_title("Dice Coefficient")
        axs[0].set_xlabel("Batch")
        axs[0].set_ylabel("Dice Score")
        axs[0].grid(True)
        axs[0].legend()

        # Accuracy
        axs[1].plot(accuracies, label="Accuracy", color="orange")
        axs[1].set_title("Accuracy")
        axs[1].set_xlabel("Batch")
        axs[1].set_ylabel("Accuracy")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )
        if self.save_plots:
            if not os.path.exists("Plots"):
                os.makedirs("Plots")
            plt.savefig(f"Plots/DistillationMetrics_{self.start_date}.png")
            plt.close()
        else:
            plt.show()

    def plot_metrics(
        self,
        dist_losses,
        train_dice_scores,
        test_dice_scores,
        train_accuracies,
        test_accuracies,
    ):
        """Plot distillation loss, Dice Score, and Accuracy."""
        epochs = list(range(1, len(dist_losses) + 1))

        plt.figure(figsize=(15, 20))

        # Distillation Loss
        plt.subplot(5, 1, 1)
        plt.plot(epochs, dist_losses, label="Distillation Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Knowledge Distillation Loss")
        plt.legend()
        plt.grid(True)

        # Dice Score train
        plt.subplot(5, 1, 2)
        plt.plot(epochs, train_dice_scores, label="Train Dice Score", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Dice Score Train")
        plt.legend()
        plt.grid(True)

        # Dice Score test
        plt.subplot(5, 1, 3)
        plt.plot(
            list(range(1, len(test_dice_scores) + 1)),
            test_dice_scores,
            label="Test Dice Score",
            color="orange",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Dice Score Test")
        plt.legend()
        plt.grid(True)

        # Accuracy Train
        plt.subplot(5, 1, 4)
        plt.plot(epochs, train_accuracies, label="Train Accuracy", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Train")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Accuracy Test
        plt.subplot(5, 1, 5)
        plt.plot(
            list(range(1, len(test_accuracies) + 1)),
            test_accuracies,
            label="Test Accuracy",
            color="red",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Test")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(f"Plots/Metrics_{self.start_date}.png")
            plt.close()
        else:
            plt.show()
