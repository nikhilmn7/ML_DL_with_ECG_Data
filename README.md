# DSC_ECG

Electrocardiograms are non-invasive detectors of electrical signals from the heart that propagate through the torso. However, data for ECGs and spatial-temporal activation maps of the heart is expensive and difficult to obtain. We used supervised machine learning techniques to accurately classify normal versus abnormal heartbeats based on ECG readings as well as the type of abnormalities that the readings reflected. We then reconstructed the spatial-temporal activation map of the heart by using deep learning techniques to create the activation potential map of each of the 75 nodes in the myocardium and their corresponding activation times.

Task 1: A binary classification problem in which we used 12-lead electrocardiogram readings to classify heartbeats as normal or abnormal.

Task 2: A multi-class classification problem in which we used 12-lead ECG readings to classify heartbeats as normal or one of four types of abnormal heartbeats.

Task 3: Using the 12-lead ECG readings, we predicted the activation time for each of the 75 nodes in the heart's myocardium. Our input was the 12 x 500 ECG reading matrix and our output was a 75 x 1 matrix represent the time in milliseconds that each node in the heart activated.

Task 4: We used the ECG data to reconstruct the spatio-temporal activation map of the heart. Our input was the 12-lead ECG reading (12 x 500 matrix, where 12 represents each lead and 500 represents the milliseconds over which the reading was taken) and our output was a 75 x 500 matrix representing the 75 nodes of the heart's myocardium, giving us a spatio-temporal reconstruction map of the heart.
