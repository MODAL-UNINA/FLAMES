import torch
import numpy as np
import os
import flwr as fl
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from loss import dice_focal_loss
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from collections import OrderedDict
from params_utils import extract_weights


class FlowerClient(fl.client.NumPyClient):

    def __init__(
        self,
        model: nn.Module,
        DEVICE: str,
        trainloader: DataLoader,
        testloader: DataLoader,
        modality: str,
        lr: float,
        num_examples: int,
        modalities_list: list,
        save_plots: bool = True,
        client_id: int = 0,
        round_epochs: int = 5,
        patience: int = 10,
        thr=0.5,
        a=0.3,
    ) -> None:

        self.model = model
        self.DEVICE = DEVICE
        self.trainloader = trainloader
        self.testloader = testloader
        self.modality = modality
        self.lr = lr
        self.round_epochs = round_epochs
        self.num_examples = num_examples
        self.loss_history = []
        self.accuracy_history = []
        self.iou_history = []
        self.dice_history = []
        self.ahd_history = []
        self.test_loss_history = []
        self.round = 0

        self.modalities_list = modalities_list
        self.save_plots = save_plots
        self.client_id = client_id
        self.patience = patience
        self.best_accuracy = 0.0
        self.early_stopping_counter = 0
        self.thr = thr
        self.patience = 20
        self.a = a


    def get_properties(self, config):
        return {
            "Modality": self.modality,
            "Client_ID": self.client_id,
            "num_examples": self.num_examples,
        }


    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def extract_parameters(self, final_parameters):
        return extract_weights(
            final_parameters, self.modalities_list, self.modality, self.client_id
        )

    def fit(self, parameters, config):
        parameters = self.extract_parameters(parameters)
        self.set_parameters(parameters)
        print("Fitting...")
        if self.round == 1:
            epochs = 10
        else:
            epochs = self.round_epochs
        self.trainloader.dataset.set_round(self.round)
        self.round += 1
        self.train(self.model, self.trainloader, epochs=epochs, lr=self.lr)
        updated_parameters = self.get_parameters(config={})
        print("len trainloader:", len(self.trainloader))
        print("len trainloader.dataset:", len(self.trainloader.dataset))
        return updated_parameters, len(self.trainloader), {}

    def train(self, model, trainloader: DataLoader, epochs=1, lr=1e-3):
        print("FIT client device:", self.DEVICE)
        print("Dice weight in loss:", self.a)
        model.train()
        model.to(self.DEVICE)
        patience_counter = 0
        best_epoch = None
        best_model_weights = model.state_dict()
        optimizer = Adam(model.parameters(), lr=lr)
        best_loss, _, _, _, _ = self.test(model, self.testloader)

        for epoch in range(epochs):
            trainloader.dataset.set_epoch(epoch)
            running_loss = 0.0
            for i, (images, masks) in enumerate(trainloader):
                images, masks = images.to(torch.float32).to(self.DEVICE), masks.to(
                    torch.float32
                ).to(self.DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = dice_focal_loss(outputs, masks, a=self.a)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i == 0: 
                    predicted_masks = outputs > self.thr  # Convert to binary mask
                    self.visualize_predictions(
                        images.cpu(),
                        masks.cpu(),
                        predicted_masks.cpu(),
                        None,
                        train=True,
                    )

            average_loss = running_loss / len(trainloader)
            self.loss_history.append(torch.tensor(average_loss))
            print(f"Epoch {epoch+1}, Loss: {average_loss}")

            loss, accuracy, iou, dice_coeff, ahd = self.test(model, self.testloader)
            self.test_loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            self.iou_history.append(iou)
            self.dice_history.append(dice_coeff)
            self.ahd_history.append(ahd)
            self.plot_accuracy_and_loss(
                self.loss_history,
                self.test_loss_history,
                self.accuracy_history,
                self.iou_history,
                self.dice_history,
                self.ahd_history,
            )
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                patience_counter = 0
                best_model_weights = model.state_dict()
                torch.save(best_model_weights, f"FL_{self.client_id}_best_model.pth")
            else:
                patience_counter += 1
                if patience_counter == self.patience:
                    if True:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        if best_epoch is not None:
            print(
                f"Best model found at epoch {best_epoch}. Restoring the best model weights."
            )
        else:
            print("No improvements were made. Restoring the previous model weights.")
        model.load_state_dict(best_model_weights)


        self.evaluate_(self.get_parameters({}))
        print("Training completed")

    def evaluate(self, parameters, config={}):
        parameters = self.extract_parameters(parameters)
        return self.evaluate_(parameters, config)

    def evaluate_(self, parameters, config={}):
        print("Evaluating...")
        self.set_parameters(parameters)
        loss, accuracy, iou, dice_coeff, ahd = self.test(self.model, self.testloader)

        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(accuracy), "IoU": float(iou)},
        )

    def test(self, net, testloader):
        """Validate the model on the test set."""
        total_pixels, correct_pixels = 0, 0
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_ahd = 0.0
        print("Testing...")

        net.eval()
        net.to(self.DEVICE)
        with torch.no_grad():
            for images, masks in testloader:
                images, masks = images.to(torch.float32).to(self.DEVICE), masks.to(
                    torch.float32
                ).to(self.DEVICE)
                outputs = net(images)

                # Calculate loss
                loss = dice_focal_loss(outputs, masks, a=self.a)
                total_loss += loss.item()

                # Calculate accuracy
                predicted_masks = outputs > 0.5  # Convert logits to binary mask
                correct_pixels += (predicted_masks == masks).sum().item()
                total_pixels += masks.numel()  # Total number of pixels

                # IoU
                total_iou += self.IoU(predicted_masks, masks)

                # Dice coefficient
                batch_dice = self.dice_coefficient(outputs, masks)
                total_dice += batch_dice

                # Hausdorff Distance
                batch_ahd = self.hausdorff_distance(outputs, masks)
                total_ahd += batch_ahd


                self.visualize_predictions(
                    images.cpu(), masks.cpu(), predicted_masks.cpu()
                )

        avg_loss = total_loss / len(testloader)
        pixel_accuracy = correct_pixels / total_pixels  # Pixel-wise accuracy
        avg_iou = total_iou / len(testloader)  # IoU medio
        avg_dice = total_dice / len(testloader)  # Coefficiente di Dice medio
        avg_ahd = total_ahd / len(testloader)  # Hausdorff Distance media
        print(f"Pixel-wise Accuracy: {pixel_accuracy * 100:.2f}%")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Dice Coefficient: {avg_dice:.4f}")
        print(f"Average Hausdorff Distance: {avg_ahd:.4f}")

        return avg_loss, pixel_accuracy, avg_iou, avg_dice, avg_ahd

    def IoU(self, predicted_masks, masks):
        epsilon = 1e-6
        # Converti i tensori in booleani
        predicted_masks = (predicted_masks > self.thr).to(
            torch.bool
        )  # Converti le probabilità in maschere binarie
        masks = (masks > self.thr).to(
            torch.bool
        )  # Assicurati che anche le maschere siano binarie

        intersection = (predicted_masks & masks).float().sum((1, 2, 3))  # Intersezione
        union = (predicted_masks | masks).float().sum((1, 2, 3))  # Unione

        iou = (intersection + epsilon) / (union + epsilon)  # Calcola IoU per batch
        total_iou = iou.mean().item()  # Calcola l'IoU medio del batch
        return total_iou

    def dice_coefficient(self, predicted_masks, masks, epsilon=1e-6):
        # Converti i tensori in booleani
        predicted_masks = (predicted_masks > 0.5).to(torch.bool)
        masks = (masks > 0.5).to(torch.bool)

        intersection = (predicted_masks & masks).float().sum((1, 2, 3))
        denominator = predicted_masks.float().sum((1, 2, 3)) + masks.float().sum(
            (1, 2, 3)
        )

        dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
        return dice.mean().item()  # Ritorna il valore medio del batch

    def hausdorff_distance(self, predicted_masks, ground_truth_masks):
        """
        Calculate the Hausdorff Distance between two sets of binary masks.

        Args:
            predicted_masks (Tensor): Predicted masks of shape (N, H, W) where N is the number of samples.
            ground_truth_masks (Tensor): Ground truth masks of shape (N, H, W).

        Returns:
            float: The average Hausdorff distance over the batch.
        """

        # Convert masks to binary
        predicted_masks = (predicted_masks > 0.5).cpu().numpy()
        ground_truth_masks = (ground_truth_masks > 0.5).cpu().numpy()

        distances = []

        for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
            # Get the coordinates of the predicted and ground truth masks
            pred_points = np.argwhere(pred_mask)
            gt_points = np.argwhere(gt_mask)

            if pred_points.size == 0 or gt_points.size == 0:
                # If one of the masks has no points, skip this mask
                distances.append(float("inf"))  # inf or a large value
                continue

            # Compute the distances between the predicted and ground truth points
            dists = cdist(pred_points, gt_points)
            hausdorff_dist = max(
                np.min(dists, axis=0).max(), np.min(dists, axis=1).max()
            )
            distances.append(hausdorff_dist)

        return np.mean(distances)  # Return the average Hausdorff distance

    def plot_loss_history(self):
        # Detach the loss history tensor from the computation graph
        loss_history_np = [loss.detach().cpu().numpy() for loss in self.loss_history]

        # plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(loss_history_np) + 1), loss_history_np)
        plt.xlabel("Epochs")
        plt.xticks(range(1, len(loss_history_np) + 1, 15))
        plt.ylabel("Loss")
        plt.title(f"Training Loss for Client ({self.client_id}_{self.modality})")
        # plt.show()

    def plot_accuracy_history(self):

        # plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(self.accuracy_history) + 1),
            self.accuracy_history,
            color="orange",
        )
        plt.xlabel("Round")
        plt.xticks(range(1, len(self.accuracy_history) + 1, 5))
        plt.ylabel("Accuracy")
        plt.title(f"Pixel-wise Accuracy for Client ({self.client_id}_ {self.modality})")
        # plt.show()

    def plot_iou_history(self):

        # plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.iou_history) + 1), self.iou_history, color="green")
        plt.xlabel("Round")
        plt.xticks(range(1, len(self.iou_history) + 1, 5))
        plt.ylabel("IoU")
        plt.title(f"IoU per Client ({self.client_id}_{self.modality})")
        # plt.show()

    def plot_test_loss_history(self):

        plt.plot(
            range(1, len(self.test_loss_history) + 1),
            self.test_loss_history,
            color="red",
        )
        plt.xlabel("Rounds")
        plt.xticks(range(1, len(self.test_loss_history) + 1, 5))
        plt.ylabel("Test Loss")
        plt.title(f"Test Loss for Client ({self.client_id}_{self.modality})")

    def plot_dice_history(self):

        plt.plot(range(1, len(self.dice_history) + 1), self.dice_history, color="blue")
        plt.xlabel("Epochs", fontsize=14)
        plt.xticks(range(1, len(self.dice_history) + 1, 10), fontsize=14)
        plt.ylabel("Dice Coefficient", fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"Dice Coefficient for Client ({self.modality})")

    def plot_ahd_history(self):
        plt.plot(range(1, len(self.ahd_history) + 1), self.ahd_history, color="purple")
        plt.xlabel("Epochs", fontsize=14)
        plt.xticks(range(1, len(self.ahd_history) + 1, 10), fontsize=14)
        plt.ylabel("Average Hausdorff Distance", fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"Average Hausdorff Distance for Client ({self.modality})")

    def plot_accuracy_and_loss(
        self,
        loss_history,
        test_loss_history,
        accuracy_history,
        iou_history,
        dice_history,
        ahd_history,
    ):
        print("Plotting...")
        plt.figure(figsize=(25, 12))

        # Plot training loss
        plt.subplot(2, 3, 1)
        self.plot_loss_history()

        # Plot test loss
        plt.subplot(2, 3, 2)
        self.plot_test_loss_history()

        # Plot accuracy
        plt.subplot(2, 3, 3)
        self.plot_accuracy_history()

        # Plot IoU
        plt.subplot(2, 3, 4)
        self.plot_iou_history()

        # Plot Dice Coefficient
        plt.subplot(2, 3, 5)
        self.plot_dice_history()

        # Plot Hausdorff Distance
        plt.subplot(2, 3, 6)
        self.plot_ahd_history()

        if self.save_plots:
            if not os.path.exists("Plots"):
                os.makedirs("Plots")
            plt.savefig(
                f"Plots/client_{self.client_id}_{self.modality}_loss_accuracy_iou_test_loss.png"
            )
            plt.close()
        else:
            plt.show()

    def plot_accuracy_vs_threshold(self, accuracies_by_threshold):
        thresholds = list(accuracies_by_threshold.keys())
        accuracies = list(accuracies_by_threshold.values())

        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, accuracies, marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Accuracy")
        plt.title(
            f"Accuracy vs Threshold for Client ({self.client_id}_{self.modality})"
        )
        plt.xticks(thresholds)

        if self.save_plots:
            if not os.path.exists("Plots_classic_FL"):
                os.makedirs("Plots_classic_FL")
            plt.savefig(
                f"Plots_classic_FL/client_{self.client_id}_{self.modality}_accuracy_vs_threshold.png"
            )
            plt.close()
        else:
            plt.show()

    def visualize_predictions(
        self, images, masks, predictions, outputs=None, train=False
    ):
        """
        Visualizza le immagini, le maschere reali e le maschere predette.
        Eventualmente, visualizza anche le probabilità di output."""
        if outputs is not None:
            fig, axs = plt.subplots(
                4, len(images), figsize=(len(images), 5), squeeze=False
            )
        else:
            fig, axs = plt.subplots(
                3, len(images), figsize=(len(images), 5), squeeze=False
            )
        if len(images) == 1:
            axs[0, 0].imshow(images[0, 0].cpu().numpy(), cmap="gray")
            axs[1, 0].imshow(masks[0, 0].cpu().numpy(), cmap="gray")
            axs[2, 0].imshow(predictions[0, 0].cpu().numpy(), cmap="gray")
            if outputs is not None:
                axs[3, 0].imshow(outputs[0, 0].cpu().numpy(), cmap="gray")
        else:
            for i in range(len(images)):
                axs[0, i].imshow(images[i, 0].cpu().numpy(), cmap="gray")
                # axs[0, i].set_title("Immagine")
                axs[1, i].imshow(masks[i, 0].cpu().numpy(), cmap="gray")
                # axs[1, i].set_title("Maschera Reale")
                axs[2, i].imshow(predictions[i, 0].cpu().numpy(), cmap="gray")
                if outputs is not None:
                    axs[3, i].imshow(outputs[i, 0].cpu().numpy(), cmap="gray")
            # axs[2, i].set_title("Maschera Predetta")

        for ax in axs:
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
        # adjust the padding between and around the subplots
        plt.tight_layout()
        if self.save_plots:
            if not os.path.exists("Plots"):
                os.makedirs("Plots")
            if train:
                plt.savefig(
                    f"Plots/client_{self.client_id}_{self.modality}_train_predictions.png"
                )
            else:
                plt.savefig(
                    f"Plots/client_{self.client_id}_{self.modality}_predictions.png"
                )
            plt.close()
        else:
            plt.show()
