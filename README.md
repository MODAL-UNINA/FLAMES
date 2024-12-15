# FLAMES-Federated-Learning-for-Advanced-Medical-Segmentation

## Abstract

Federated Learning (FL) is gaining traction across numerous fields for its ability to foster collaboration among multiple participants while preserving data privacy. In the medical domain, FL plays a pivotal role in collaboratively training models safeguarding patient privacy, as it enables institutions to share knowledge learned while maintaining control over their data, which often varies in modality, source, and quantity. Institutions are often specialized in the treatment of one or a few types of tumors, typically focusing on a specific organ. Hence, different institutions may contribute with distinct types of medical imaging data of various organs, originating from diverse machines and patient populations. Collaboration among these institutions enhances performance on shared tasks across different areas of the body.\\
The framework employs modality-specific models hosted on the server, each designed for a particular imaging modality. Each modality-specific model on the server is designed to predict the presence of tumors in scans from its respective modality, regardless of the organ being imaged. Clients focus on their specific imaging modality, utilizing knowledge derived from images contributed by institutions employing the same modality, even when these images pertain to different organs. This approach facilitates broader collaboration, extending beyond institutions specializing in the same organ to include those working within the same imaging modality. This approach also helps avoid the introduction of potential noise from clients with images of different modalities, which might hinder the model's ability to effectively specialize and adapt to the data specific to each institution.
Experiments showed that FLAMES achieves strong performance on server data, even when tested across different organs, demonstrating its ability to generalize effectively across diverse medical imaging datasets.

#

![alt text](https://github.com/MODAL-UNINA/FLAMES---Federated-Learning-for-Advanced-Medical-Segmentation/blob/main/Images/framework_4.png)


### Acknowledgments
This work has been supported by 
- PNRR Centro Nazionale HPC, Big Data e Quantum Computing, (CN 00000013)(CUP: E63C22000980007), under the NRRP MUR program funded by the NextGenerationEU, Innovation Funds "SESG - Integrated Platform for Enhanced Analysis of Environmental, Social, and Governance".
- PNRR project FAIR -  Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU.
- G.A.N.D.A.L.F. - Gan Approaches for Non-iiD Aiding Learning in Federations, CUP: E53D23008290006, PNRR - Missione 4 “Istruzione e Ricerca” - Componente C2 Investimento 1.1 “Fondo per il Programma Nazionale di Ricerca e Progetti di Rilevante Interesse Nazionale (PRIN)”.
