# %%

import numpy as np
from flwr.common.typing import NDArrays
from flwr.common.logger import log
from logging import INFO


def gen_final_parameters(aggregated_weights_by_modality, server_round):
    list_parameters = []
    list_total = []
    list_client_ids = []
    n_modalities = np.array([len(aggregated_weights_by_modality.keys())])
    list_total.append(n_modalities)

    assert all(
        len(q)
        == len(list(list(aggregated_weights_by_modality.values())[0].values())[0])
        for v in aggregated_weights_by_modality.values()
        for q in v.values()
    ), "Different number of parameters in the models"

    len_weights = np.array(
        [len(list(list(aggregated_weights_by_modality.values())[0].values())[0])]
    )

    for modality, weights_modality in aggregated_weights_by_modality.items():
        log(INFO, f"Round {server_round} - Modality {modality}")
        modality_clients = []
        for i, (client_id, weights) in enumerate((weights_modality.items())):
            log(INFO, f"Round {server_round} - Client {client_id}")

            modality_clients.append(client_id)
            list_parameters.extend(weights)

        modality_clients = np.array(modality_clients)
        list_client_ids.append(modality_clients)

    list_total.extend(list_client_ids)
    list_total.append(len_weights)

    list_total.extend(list_parameters)

    return list_total


def extract_weights(
    final_parameters: NDArrays,
    modalities_list: list[str],
    modality: str,
    client_id: int,
):
    log(INFO, f"Extracting parameters for client_id: {client_id}")
    number_of_modalities = len(modalities_list)

    i_ptr = 0
    n_modalities = final_parameters[i_ptr][0]
    assert (
        n_modalities == number_of_modalities
    ), f"Number of modalities mismatch: {n_modalities} != {number_of_modalities}"

    i_ptr += 1

    client_ids_all = final_parameters[i_ptr : n_modalities + i_ptr]
    log(INFO, f"Number of clients for all modalities: {client_ids_all}")

    i_ptr += n_modalities

    len_weight = final_parameters[i_ptr][0]
    log(INFO, f"len_of_params_all: {len_weight}")

    i_ptr += 1

    idx_modality = modalities_list.index(modality)
    client_ids = client_ids_all[idx_modality]
    log(INFO, f"Number of clients for modality: {client_ids}")

    idx_client_id = np.where(client_ids == client_id)[0][0]
    log(INFO, f"Index of client_id: {idx_client_id}")
    log(INFO, "Offset search")

    i_modality = 0
    while i_modality < idx_modality:
        client_ids_i_modality = client_ids_all[i_modality]
        i_ptr += len_weight * len(client_ids_i_modality)
        log(
            INFO,
            f"modality: {modalities_list[i_modality]} - client ids i modality {client_ids_i_modality}"
            f" - Offset(add) {len_weight * len(client_ids_i_modality)}",
        )
        i_modality += 1

    i_ptr += len_weight * idx_client_id
    log(INFO, f"Offset: {i_ptr}")

    parameters_client = final_parameters[i_ptr : i_ptr + len_weight]
    log(
        INFO,
        f"Extracted parameters for modality: {modality} - Type: {type(parameters_client)}",
    )
    return parameters_client


# %%
