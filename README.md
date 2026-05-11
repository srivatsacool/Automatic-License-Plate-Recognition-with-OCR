# Automatic License Plate Recognition with OCR (ALPR + OCR)

<p align="center">
  <img src="https://user-gen-media-assets.s3.amazonaws.com/gpt4o_images/e52e064d-2c23-4e5a-b3ee-f2fa5704d3f9.png" alt="ALPR Hero Banner" width="100%" />
</p>

## 1. Introduction

Automatic License Plate Recognition (ALPR) is a computer vision system designed to automatically detect vehicle license plates and extract the alphanumeric characters from them. This project, **Automatic License Plate Recognition with OCR**, demonstrates a complete ALPR pipeline implemented as a web application.

The system is capable of processing both **images and videos**, detecting license plates, and extracting the text using Optical Character Recognition (OCR). The application is built using **Python**, **OpenCV**, **EasyOCR**, and **Streamlit** for the user interface.

---

## Demo :
Try Link - https://srivatsacool-combine-1--license-plate-detection-with-ocr-abxfk4.streamlit.app/

On Image :

<p align="center">
  <img src="https://user-images.githubusercontent.com/76219802/214331437-c8b24ea6-270c-4126-af27-3101ab53a897.png" />
</p>

<video src="https://user-images.githubusercontent.com/76219802/214332571-d343d046-4289-40e0-a576-1cb2f922749b.mp4" controls="controls" style="max-width: 1000px;" autoplay = "autoplay">
</video>

On Video :

<video src="https://user-images.githubusercontent.com/76219802/214332706-f53988c8-1ea3-4450-b497-8db703b1842a.mp4" controls="controls" style="max-width: 1000px;" autoplay = "autoplay">
</video>

---

## 2. Project Objectives

The primary objectives of this project are:

- Detect license plates from vehicle images and videos
- Extract readable text from detected license plates
- Provide a simple web-based interface for interaction
- Demonstrate a practical real-world application of computer vision and OCR
- Enable future use cases such as traffic monitoring and law enforcement assistance

---

## 3. Repository Overview

**GitHub Repository:**  
https://github.com/srivatsacool/Automatic-License-Plate-Recognition-with-OCR

### Repository Structure

```text
Automatic-License-Plate-Recognition-with-OCR/
├── app.py
├── requirements.txt
├── README.md
├── sample images (.jpg)
├── sample videos (.mp4)
└── display assets
```

* `app.py` – Main Streamlit application
* `requirements.txt` – Python dependencies
* Sample media files for testing and demonstration

---

## 4. Technology Stack

### Programming Language

* Python 3

### Libraries and Frameworks

| Library                    | Purpose                        |
| -------------------------- | ------------------------------ |
| OpenCV                     | Image and video processing     |
| EasyOCR                    | Optical Character Recognition  |
| NumPy                      | Numerical and array operations |
| Streamlit                  | Web application interface      |
| Pillow                     | Image handling                 |
| imutils                    | Image utility functions        |
| onnxruntime-gpu (optional) | Model inference acceleration   |

---

## 5. System Architecture

The ALPR system follows a standard multi-stage computer vision pipeline.

### High-Level Workflow

1. User uploads an image or video
2. Video frames are extracted (if applicable)
3. License plate regions are detected
4. Detected plates are cropped
5. OCR is applied to cropped plates
6. Extracted text is displayed in the UI

### Workflow Diagram (Logical)

```text
Input (Image / Video)
        ↓
Preprocessing
        ↓
License Plate Detection
        ↓
Plate Cropping
        ↓
OCR (Text Extraction)
        ↓
Result Display (UI)
```

---

## 6. License Plate Detection

The detection module identifies regions in an image or frame that correspond to vehicle license plates. While the repository does not publicly include the trained detection model, the pipeline follows common ALPR practices:

* Image resizing and normalization
* Object detection using a trained model
* Bounding box extraction
* Region of Interest (ROI) cropping

> Note: The trained detection model must be obtained separately from the project author.

---

## 7. Optical Character Recognition (OCR)

After the license plate region is detected and cropped, **EasyOCR** is used to extract alphanumeric characters.

### Why EasyOCR?

* Supports multiple languages
* Handles noisy and low-resolution text
* No complex training required
* Well-suited for license plate formats

The OCR output is post-processed to improve readability and remove unwanted characters.

---

## 8. Streamlit Web Application

The project uses **Streamlit** to provide an interactive web interface.

### UI Features

* Image upload
* Video upload
* Light and dark mode toggle
* Real-time visualization
* Display of detected plates and extracted text

### Running the Application

```bash
streamlit run app.py
```

---

## 9. Installation and Setup

### Prerequisites

* Python 3.8+
* pip package manager

### Installation Steps

```bash
git clone https://github.com/srivatsacool/Automatic-License-Plate-Recognition-with-OCR
cd Automatic-License-Plate-Recognition-with-OCR
pip install -r requirements.txt
```

---

## 10. Use Cases

* Traffic law enforcement
* Speed violation monitoring
* Automated toll systems
* Parking management
* Smart city surveillance systems

---

## 11. Limitations

* Detection model is not included publicly
* OCR accuracy depends on lighting and image quality
* Performance may degrade with angled or blurred plates
* Limited handling of non-standard license plate formats

---

## 12. Future Enhancements

* Include an open-source detection model
* Improve OCR accuracy with preprocessing
* Add real-time camera feed support
* Store detected plates in a database
* Integrate alert or notification systems
* Deploy as a cloud-based service

---

## 13. Conclusion

The **Automatic License Plate Recognition with OCR** project successfully demonstrates a full ALPR pipeline using modern computer vision tools. By combining detection, OCR, and a web-based interface, the project provides a strong foundation for real-world intelligent traffic and surveillance systems.

The modular design allows easy enhancement and integration into larger smart city or transportation platforms.

---

## 14. References

* GitHub Repository: [https://github.com/srivatsacool/Automatic-License-Plate-Recognition-with-OCR](https://github.com/srivatsacool/Automatic-License-Plate-Recognition-with-OCR)
* Optical Character Recognition (OCR): [https://en.wikipedia.org/wiki/Optical_character_recognition](https://en.wikipedia.org/wiki/Optical_character_recognition)
* License Plate Recognition Overview: [https://www.geeksforgeeks.org/automatic-license-number-plate-recognition-system/](https://www.geeksforgeeks.org/automatic-license-number-plate-recognition-system/)
