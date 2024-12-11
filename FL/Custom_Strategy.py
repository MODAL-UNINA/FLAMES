import flwr as fl
import numpy as np
from logging import WARN
from functools import reduce
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import List, Tuple, Optional, Dict, Union
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    NDArrays,
    GetPropertiesIns,
)
from params_utils import gen_final_parameters

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


""" The server maintains a model for each image modality.
Models are trained in parallel. The server aggregates only the weights received from clients
with the same modality of the model and returns updated weights to the same clients"""


class CustomStrategy(fl.server.strategy.FedAvg):
    """Custom Federated Averaging strategy."""

    def __init__(
        self,
        strategies: Dict[str, fl.server.strategy.FedAvg],
        min_num_clients: Optional[int] = 3,
        initial_parameters: Optional[Parameters] = None,
    ):
        super().__init__()
        self.strategies = strategies
        self.min_num_clients = min_num_clients
        self.loss_history = {
            mod: [] for mod in self.strategies.keys()
        }
        self.initial_parameters = initial_parameters
        self.min_num_clients = min_num_clients

    def client_properties(self, client_manager: ClientManager):
        client_manager.wait_for(num_clients=self.min_num_clients)
        client_properties = {}
        for cid, client in client_manager.all().items():
            ins = GetPropertiesIns(config={})
            properties_res = client.get_properties(ins, timeout=None, group_id=None)
            client_properties[cid] = properties_res.properties
        return client_properties

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        clients_per_modality: Optional[Dict[str, List[Tuple[str, int]]]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggrega i risultati di 'fit' usando una media pesata personalizzata per ogni modalità di client."""

        if not results:
            print("No results", server_round)
            return None, {}

        if not self.accept_failures and failures:
            print("Failures")
            return None, {}

        aggregated_weights_by_modality = {}
        weights_shapes_by_modality = {}
        weigths_server_by_modality = {}
        results.sort(key=lambda x: x[0].cid)

        for modality in clients_per_modality.keys():
            client_ids = clients_per_modality[modality]
            cids_map = dict(client_ids)
            modality_results = [
                (client, fit_res)
                for client, fit_res in results
                if client.cid in cids_map.keys()
            ]

            if not modality_results:
                continue
            modality_results = [
                (client, cids_map[client.cid], fit_res)
                for client, fit_res in modality_results
            ]
            aggregated_weights_by_modality[modality] = {}
            weights_shapes_by_modality[modality] = {}

            all_weights = []

            for client, client_id, fit_res in modality_results:
                client_weights = parameters_to_ndarrays(fit_res.parameters)
                num_examples = fit_res.num_examples

                all_weights.append(
                    (client.cid, client_id, client_weights, num_examples)
                )

            weigths_server_by_modality[modality] = [
                np.mean(np.stack([z[2][i] for z in all_weights], axis=0), axis=0)
                for i in range(len(all_weights[0][2]))
            ]
            for client, client_id, fit_res in modality_results:
                client_weights = parameters_to_ndarrays(fit_res.parameters)
                num_examples = fit_res.num_examples

                alpha = 0.1 + (1 / len(client_ids))
                print(f"client {client_id} --> alpha= {alpha}")

                aggregated_weights_by_modality[modality][client_id] = self.aggregate(
                    all_weights, client_id, alpha
                )

            aggregated_weights_by_modality[modality] = {
                k: aggregated_weights_by_modality[modality][k]
                for k in sorted(aggregated_weights_by_modality[modality].keys())
            }

        aggregated_weights_by_modality = {
            k: aggregated_weights_by_modality[k]
            for k in sorted(aggregated_weights_by_modality.keys())
        }
        weigths_server_by_modality = {
            k: weigths_server_by_modality[k]
            for k in sorted(weigths_server_by_modality.keys())
        }
        final_parameters = gen_final_parameters(
            aggregated_weights_by_modality, server_round
        )

        final_parameters = ndarrays_to_parameters(final_parameters)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARN, "No fit_metrics_aggregation_fn provided")

        return final_parameters, metrics_aggregated, weigths_server_by_modality

    def aggregate(
        self, all_weights: List[Tuple[str, NDArrays, int]], client_id: str, alpha: float
    ) -> NDArrays:
        """Compute weighted average, giving more weight to the local client."""


        local_weight = alpha
        other_weights = []

        inv_num_samples = [
            1.0 / num_examples if num_examples > 0 else 0
            for (_, id, _, num_examples) in all_weights
            if id != client_id
        ]

        total_inv_samples = sum(inv_num_samples)

        if total_inv_samples > 0:
            remaining_weight = 1 - local_weight
            other_weights = [
                (remaining_weight * (inv_sample / total_inv_samples))
                for inv_sample in inv_num_samples
            ]
        else:
            other_weights = [0] * (
                len(all_weights) - 1
            )
        weighted_weights = []

        print(f"client_id: {client_id}")
        local_weights = next(
            weights for _, client, weights, _ in all_weights if client == client_id
        )

        weighted_weights.append(
            [layer * local_weight for layer in local_weights]
        )

        other_clients = [
            (client, weights, num_examples)
            for _, client, weights, num_examples in all_weights
            if client != client_id
        ]

        for (client, weights, num_examples), weight in zip(
            other_clients, other_weights
        ):
            weighted_weights.append(
                [layer * weight for layer in weights]
            )

        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
