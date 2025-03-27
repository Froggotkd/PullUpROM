# 🏋️ Rep Counter  

A computer vision-based rep counter that tracks ideal range of motion (ROM) for exercises like pull-ups and bicep curls using OpenCV and MediaPipe.  

## 📌 How It Works  

- **Pull-Ups:** The rep is counted when:  
  - Arms are fully extended at the bottom.  
  - Chin passes over the bar at the top.  

- **Bicep Curls:** The rep is counted when:  
  - The angle between the wrist, elbow, and shoulder is **greater than 160°** at full extension.  
  - The angle decreases **below 30°** at peak contraction.  

## 🛠️ Tech Stack  

- **OpenCV** – For image processing.  
- **MediaPipe** – For pose estimation and motion tracking.  
- **Flask** – For serving the model via an API.  

![image](https://github.com/user-attachments/assets/d6c5304c-5828-4374-a529-0e3af5649464)
![image](https://github.com/user-attachments/assets/4117acaa-5d7c-4c59-b342-5a80c9c1b798)
![image](https://github.com/user-attachments/assets/2d679098-4c84-4390-8b54-206fda774d22)
