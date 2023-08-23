import tensorflow as tf
from typing import List
import cv2
import os

# Vocabulary list, consisting of the given characters
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Mapping from characters to integers
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Mapping from integers back to characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path:str) -> List[float]: 
    """
    Load a video from the given path, convert frames to grayscale and normalize them.
        Parameters:
        - path (str): Path to the video file.

        Returns:
        - List[float]: A list of normalized grayscale frames from the video.
    """

    cap = cv2.VideoCapture(path)
    frames = []
    
    # Read each frame from the video
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        
        # Crop the frame
        frames.append(frame[190:236,80:220,:])
    
    # Release the video file
    cap.release()
    
    # Normalize the frames using mean and standard deviation
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    

def load_alignments(path:str) -> List[str]: 
    """
    Load alignments from the given path and tokenize them.
        Parameters:
        - path (str): Path to the alignments file.

        Returns:
        - List[str]: A list of tokenized alignments.
    """

    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    
    # Tokenize each line of the file
    for line in lines:
        line = line.split()
        
        # Exclude 'sil' tokens and add characters to the tokens list
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path: str): 
    """
    Load video and its associated alignments from the given path.
        Parameters:
        - path (str): Path to the video or alignment file.

        Returns:
        - Tuple[List[float], List[str]]: A tuple containing a list of normalized grayscale frames and a list of tokenized alignments.
    """
    

    # Convert byte path to string
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0] # for Mac
    
    # Construct paths for video and alignment files
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    
    # Load the video frames and alignments
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments