import cv2
import base64
from dotenv import load_dotenv

load_dotenv()

def capture_image()->str:
    """
    Captures one frame from the default webcam, resize it, 
    encodes it into base64 JPEG (raw string ) and returns it.
    """

    # it tries 4 times to open the webcam to capture
    for idx in range(4):
        cap= cv2.VideoCapture(idx, cv2.CAP_DSHOW) #for macOS cv2.CAP_AVFOUNDATION
        if cap.isOpened():
            print("cap is opened !!")
            for _ in range(10): #kind of warm up
                cap.read() # we are taking the last frame
            ret, frame = cap.read()

            cap.release()
            if not ret:
                continue
            cv2.imwrite("sample.jpg",frame) #for testing 
            ret,buf = cv2.imencode('.jpg', frame)

            if ret:
                return base64.b64encode(buf).decode('utf-8')
    raise RuntimeError("Could not open any webcam ( tried 3 times)")

from groq import Groq

def analyse_image_with_query(query:str)->str:
    """
     Expects a string 'query'
     Captures the image and sends the query and image to the 
     Groq's vision chat API and returns the analysis.
    """
    img_b64 = capture_image() #returns the base64 string of the image captured
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"

    if not query and not img_b64:
        return "Error: both 'query' and 'image' fields required"
    
    client = Groq()
    messages=[
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":query
                },
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:image/jpeg;base64,{img_b64}",
                    },
                }
            ]
        }
    ]
    client.chat
    chat_completion = client.chat.completions.create(
        messages = messages,
        model = model
    )

    return chat_completion.choices[0].message.content

# for testing this file standalone -->  uv run tools.py
#query = "Is the person wearing spectale"
#print(analyse_image_with_query(query))