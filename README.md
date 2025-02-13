**PPG based Noninvasive Blood Glucose Monitoring using Multi-view Attention and Cascaded BiLSTM Hierarchical Feature Fusion Approach**

We developed a noninvasive BG monitoring approach on photoplethysmography (PPG) signals using multi-view attention and a cascaded BiLSTM hierarchical feature fusion approach. 
1. Firstly, we implemented a convolutional multi-view attention block to extract the temporal features through adaptive contextual information aggregation. 
2. Secondly, we built a cascaded BiLSTM network to efficiently extract the fine-grained features through bidirectional learning. 
3. Finally, we developed a hierarchical feature fusion with bilinear polling through cross-layer interaction to obtain higher-order features for BG monitoring. 

For validation, we conducted comprehensive experimentation on up to 6 days of PPG and BG data from 21 participants. The proposed approach showed competitive results compared to existing approaches by RMSE of 1.67 mmol/L and MARD of 17.88%. Additionally, the clinical accuracy using Clarke error grid (CEG) analysis showed 98.80% of BG values in Zone A+B. Therefore, the proposed approach offers a favorable solution in diabetes management by noninvasively monitoring the BG levels.

Published in "IEEE Journal of Biomedical and Health Informatics": https://ieeexplore.ieee.org/document/10684156
