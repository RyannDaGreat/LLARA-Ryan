import rp
from PIL import Image
import random
import re
import requests
from io import BytesIO
import cv2
import numpy as np

import fire

url = "http://130.245.125.22:30000/chat"  # Replace with your server URL

img_w = 640
img_h = 320
# y_offset = 80 # crop for vima
y_offset = 0 # crop for vima

# TEST
questions = [
    "Could you write down what needs to be done to complete the task on this scene?",
    "List out the actions needed to accomplish the task in this scene.",
    "What actions are necessary to perform the task on this scene?",
    "Can you describe what needs to be done on this scene to complete the task?",
    "What steps are required to perform the task shown in this scene?",
    "List the actions needed to perform the task given below.",
    "On the following scene, could you list what actions are required to perform the task?",
    "Describe what actions are needed on this scene to complete the task.",
    "What do you need to do on this scene to accomplish the task?",
    "List the actions required to perform the task given on this scene.",
    "Could you please describe the steps needed to perform the task on this scene?",
    "Write down the actions required to perform the task on this scene.",
    "Please write down the actions required to perform the task shown below.",
    "Can you explain what needs to be done to perform the task in this scene?",
    "Describe the actions required to complete the task on this scene.",
]

# Convert the PIL image to a byte stream
def convert_pil_image_to_bytes(image):
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')  # Change format if needed
    byte_stream.seek(0)
    return byte_stream

def image_qa(image, question):
    # Convert the PIL image to a byte stream
    image_bytes = convert_pil_image_to_bytes(image)
    # Define the payload
    files = {
        'image': ('image.png', image_bytes, 'image/png'),
        'text': (None, question)  # No filename or content type needed for plain text
    }

    # Send the POST request
    response = requests.post(url, files=files)
    return response.text

def prepare_prompt(p:str) -> str:
    task_prompt = f'<task>{p}</task>'
    user_prompt = random.choice(questions)
    
    format_prompt = "Every action you take must include two locations in the format of <b>(x, y)</b> and one clockwise rotation angle in the format of <r>[r]</r>. "
    format_prompt += "The first location is the image coordinate where you use a suction cup to pick up the object, and the second location is where you place the object."
    format_prompt += "The image coordinate ranges from 0 to 1. The rotation angle indicates how many degrees you rotate the object clockwise, and it ranges from -359 to 359."
    
    return '\n'.join(['<image>', task_prompt, user_prompt, format_prompt])


def parse_coor(s: str):
    # remove ( )
    l = s[1:-1].split(',')
    assert len(l) == 2
    px, py = float(l[0]), float(l[1])
    px *= img_w # image size
    py *= img_h
    return int(px), int(py + y_offset)

def action_to_text(action: dict) -> str:
    w, h = img_w, img_h
    px = action['pick'][0]
    py = action['pick'][1] - y_offset
    tx = action['place'][0]
    ty = action['place'][1] - y_offset
    rotation = action['rotation']
    return f'Pick up the object at <b>({px / w :.3f}, {py / h: .3f})</b>, rotate <r>[{-rotation}]</r> degrees, and drop it at <b>({tx / w :.3f}, {ty / h :.3f})</b>.'
        

def get_vima_img(img):
    if isinstance(img,str):
        img = rp.load_image(img)
    img=rp.resize_image_to_hold(img,height=320,width=640)
    img=rp.crop_image(img,height=320,width=640,origin='center')
    img = rp.as_pil_image(rp.as_byte_image(rp.as_rgb_image(img)))
    return img


def run_llara(
    img='/Users/ryan/Downloads/12_last.png',
    task_prompt="Put the object on top of another object.",
):
    """This is one of the critical functions."""
    action_queue = []
    # task_prompt = 'Put the object on top of another object.'
    prompts_to_vlm = prepare_prompt(task_prompt)
    
    # vima_img = Image.open('/Users/ryan/Downloads/12_last.png').crop((0, y_offset, 640, 320 + y_offset))
    vima_img=get_vima_img(img)
    ans = image_qa(image=vima_img, question=prompts_to_vlm)
    
    str_actions = re.findall(f'\(.+?,.+?\)', ans)
    str_rotation = re.findall(f'\[(-*\d*)\]', ans)
    m_len = min(len(str_actions) // 2, len(str_rotation))
    action_queue.extend([str_actions[idx * 2: idx * 2 + 2] + str_rotation[idx: idx + 1]  for idx in range(m_len)])
    # perform the action from VLM
    if len(action_queue) == 0:
        print('No action in the queue.')
    else:
        pick_place_point = action_queue.pop(0)
        pick = parse_coor(pick_place_point[0])
        place = parse_coor(pick_place_point[1])
        rotation = -int(pick_place_point[2])
    
        print(pick, place, rotation) 
        # pick is a point in pixel coordinates (x, y)
        # place is another point
        # the default image size should be 640 x 320

    frame = np.array(vima_img)
    if pick and place:
        frame = cv2.arrowedLine(frame, pick, place, (0, 0, 255), 1)
    if pick:
        frame = cv2.circle(frame, pick, 3, (255, 0, 0), 2)
    if place:
        frame = cv2.circle(frame, place, 3, (0, 255, 0), 2)

    # rp.display_image(frame,1)

    return rp.gather_vars('frame pick place prompts_to_vlm')

def demo():
    input_images = [
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/1_last.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/3_last.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/4_first.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/7_last.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/9_first.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/10_first.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/12_first.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/15_first.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/18_last.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/18_first.png',
        '/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_dinbcauxc_ftxarmact_task1/19_first.png',
    ]


    task_prompts=[
        'Put the small thing in the pan',
        'Put the small thing in the plate',
        'Put the small thing in the dish',
        'grab the small object and put it in the pan',
        'grab the small object and put it in the plate',
        'grab the small object and put it in the dish',
        'move the small object into the pan',
        'move the small object into the plate',
        'move the small object into the dish',
    ]

    grid=[]

    input_images=rp.shuffled(input_images)
    task_prompts=rp.shuffled(task_prompts)
    
    for task_prompt in rp.eta(task_prompts, title='task_prompt'):
        for input_image in rp.eta(input_images, title='input_image'):
            output = run_llara(input_image,task_prompt)
            rp.display_image(output.frame)









if __name__ == '__main__':
    demo()
    # fire.Fire()
    # output=main('/Users/ryan/Downloads/12_last.png','Put the yellow corn on the brown thing')
    # rp.display_image(output.frame,1)
