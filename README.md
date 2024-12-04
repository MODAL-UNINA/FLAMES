# FLAMES-Federated-Learning-for-Advanced-Medical-Segmentation

## Abstract

Federated Learning (FL) is gaining traction across numerous fields for its ability to foster collaboration among multiple participants while preserving data privacy. In the medical domain, FL plays a pivotal role in collaboratively train models safeguarding patient privacy, as it enables institutions to share knowledge learned while maintaining control over their data, which often varies in modality, source, and quantity. For example, different institutions may contribute with distinct types of medical imaging data, originating from diverse machines and patient populations. Collaboration among these institutions enhances performance on shared tasks by leveraging the diversity of modalities across sites.
The framework employs several modality-specific models hosted on the server, each designed for a particular imaging modality, alongside a global model. Through knowledge distillation (KD), the global model transfers knowledge to a smaller, unified model that manages all modalities. This approach enhances generalization across modalities, ensuring diagnostic precision regardless of the input image type. By collaborating with other centers, each client benefits from shared insights while retaining specialization in its own data modality.

#

![alt text](https://github.com/MODAL-UNINA/FLAMES---Federated-Learning-for-Advanced-Medical-Segmentation/blob/main/Images/Framework.png)


### Acknowledgments
This work has been supported by PNRR Centro Nazionale HPC, Big Data e Quantum
Computing, (CN 00000013)(CUP: E63C22000980007), under the NRRP MUR program funded by the NextGenerationEU, Innovation Funds "SESG - Integrated Platform for Enhanced Analysis of Environmental, Social, and Governance".
