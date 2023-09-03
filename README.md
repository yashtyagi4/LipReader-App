![Screenshot 2023-09-03 at 1 19 59 AM](https://github.com/yashtyagi4/LipReader-App/assets/85970478/76f5f58a-421e-4ed7-adfc-0cb30448083b)

<br>

Sheldon is an advanced deep-learning **model capable of deciphering lip movements from silent videos** of individuals speaking. This solution is built using a sophisticated end-to-end Keras model leveraging spatiotemporal C3D (3D Convolutional Neural Network) for video preprocessing and incorporating Bi-directional LSTM with CTC output handling to precisely decode speech from muted videos.

In addition to its deep learning capabilities, Sheldon has been transformed into an intuitive web application to showcase the model's prowess. This application is developed using Streamlit, enhanced with custom HTML and CSS to deliver a seamless user interface and experience.

<br>

## Demonstration
https://github.com/yashtyagi4/LipReader-App/assets/85970478/f726333c-f9cc-495c-b53e-82dd88233635

<br>

## Dependencies
Sheldon has a range of dependencies essential for its operation. Here's a breakdown:

#### Python Libraries:
-   **OS**: For interacting with the operating system.
-   **BytesIO**: To handle byte-like objects (used primarily when the app is dealing with uploaded files).
-   **Base64**: To encode the binary data like images or video files.
-   **Typing**: For support with type hints in the codebase.

#### Deep Learning:
-   **TensorFlow**: The core framework that supports our Keras model.
-   **Keras**: The high-level neural networks API, written in Python and capable of running on top of TensorFlow.
-   **3D Convolutional Neural Network (C3N), Long Short-Term Memory (LSTM), Connectionist temporal classification (CTC)**: Various deep learning architectures and methodologies employed in the model.

#### Web Application:
-   **Streamlit**: An open-source app framework for Machine Learning and Data Science teams.
-   **HTML & CSS**: Used for the enhancement of user interface and experience in the web application.

#### Image and Video Processing:
-   **OpenCV (cv2)**: Open-source computer vision and machine learning software library.
-   **Python Imaging Library (PIL)**: For opening, manipulating, and saving many different image file formats.
-   **ImageIO**: Provides an easy interface to read and write a wide range of image data.

<br>

## Setting Up and Running

### Prerequisites:

-   Python 3.6 or higher

### Installation:

1.  Clone the repository:

	**`git clone https://github.com/yashtyagi4/LipReader-App.git`** 
	**`cd LipReader-App`** 

2.  Set up a virtual environment (optional but recommended):

	**``python -m venv venv` ``** 

	On Mac:
		**``source venv/bin/activate``**

	On Windows:
		**``source venv\Scripts\activate``**

3.  Install the required dependencies:

	**`pip install tensorflow streamlit opencv-python imageio Pillow`** 

4.  Run the Streamlit app:

	**`streamlit run app.py`** 

Once you run the above command, Streamlit should open a new tab in your default web browser with the application.

<br>

## License

This project is under the MIT license. For more information, please refer to the `LICENSE` file in the repository.
