import rp
from PIL import Image
import random
import re
import requests
from io import BytesIO
import cv2
import numpy as np

import fire

urls = [
    "http://130.245.125.22:30000/chat",
    "http://130.245.125.22:30001/chat",
    "http://130.245.125.22:30002/chat",
    "http://130.245.125.22:30003/chat",
]
url_index = 0
url = urls[url_index % len(urls)]

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
    if pick:
        frame = cv2.circle(frame, pick, 3, (255, 0, 0), 10,cv2.LINE_AA)
    if place:
        frame = cv2.circle(frame, place, 3, (0, 255, 0), 10,cv2.LINE_AA)
    if pick and place:
        # Anti-aliased arrow with smoother edges
        frame = cv2.arrowedLine(frame, pick, place, (255, 255, 255), 3, cv2.LINE_AA)  # Outer white line
        frame = cv2.arrowedLine(frame, pick, place, (0, 0, 0), 1, cv2.LINE_AA)        # Inner black line

    # rp.display_image(frame,1)

    return rp.gather_vars('frame pick place prompts_to_vlm')

import numpy as np

def gridlines_image(width, height, foreground_color=(1.,1.,1.,1.), background_color=(0.,0.,0.,0.), line_width=1, spacing=32):
    """
    Generate a grid pattern as an RGBA numpy array.
    
    Args:
        width (int): Width of the output image in pixels
        height (int): Height of the output image in pixels
        foreground_color (tuple): RGBA values for grid lines (0-1 range)
        background_color (tuple): RGBA values for background (0-1 range)
        line_width (int): Width of grid lines in pixels
        spacing (int): Number of pixels between grid lines
        
    Returns:
        numpy.ndarray: RGBA image array with shape (height, width, 4)
    """
    # Ensure colors are numpy arrays with float values
    fg_color = np.array(foreground_color, dtype=np.float32)
    bg_color = np.array(background_color, dtype=np.float32)
    
    # Create base image
    image = np.full((height, width, 4), bg_color, dtype=np.float32)
    
    # Draw vertical lines
    for x in range(spacing, width, spacing):
        x_start = x - line_width//2
        x_end = x + line_width//2 + line_width%2
        if x_start < width:  # Check if line start is within image bounds
            x_end = min(x_end, width)  # Ensure we don't draw outside image
            image[:, x_start:x_end] = fg_color
    
    # Draw horizontal lines
    for y in range(spacing, height, spacing):
        y_start = y - line_width//2
        y_end = y + line_width//2 + line_width%2
        if y_start < height:  # Check if line start is within image bounds
            y_end = min(y_end, height)  # Ensure we don't draw outside image
            image[y_start:y_end, :] = fg_color
    
    return image

def with_gridlines(image):
    gridlines=gridlines_image(rp.get_image_width(image),rp.get_image_height(image))
    image=rp.blend_images(image,gridlines,.5)
    return image

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
    task_prompts = [
        "Never gonna give you up",
        "Never gonna let you down",
        "put the thing",
        "put the small object to the right of the dish",
        "put the small object to the top of the dish",
        "put the small object in the dish",
        "put the dish on the small object",
        "drop the dish on the other thing",
    ]
    # input_images=rp.random_batch(input_images, 8)


    input_images = [
        "/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_inbc_auxd_feedme/4_last.png",
        "/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_inbc_auxd_feedme/4_first.png",
        # "/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_inbc_auxd_feedme/5_last.png",
        "/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_inbc_auxd_feedme/2_first.png",
        "/Users/ryan/Downloads/_xArm-VIMA/iclr_rebuttal_episodes_inbc_auxd_feedme/5_first.png",
    ]
    task_prompts=[
        # 'put the red thing on the yellow thing',
        # 'put the pepper on the duck',
        # 'put the donut on the pepper',
        # 'put the duck on the pepper',
        # 'put the donut on the duck',
        # 'irritate the donut by moving other food on it',
        # 'feed the duck healthy food',
        # 'put the thing that swims next to the thing thats red',
        # 'feed the duck something unhealthy',
        # 'eat the duck',
        # 'eat the pepper',
        # 'move the donut',
        # 'I have a yellow duck named Bill and a pink donut named Bob and a red pepper named Mary. Put Mary on Bob.',
        # 'I have a yellow duck named Bill and a pink donut named Bob and a red pepper named Mary. Put Bob on Mary.',
        # 'I have a yellow duck named Bill and a pink donut named Bob and a red pepper named Mary. Put Bill on Bob.',
        # 'I have a yellow duck named Bill and a pink donut named Bob and a red pepper named Mary. Put Bob on Bill.',
        # 'I have a yellow duck named Bill and a pink donut named Bob and a red pepper named Mary. Put Bill on Mary.',
        # 'I have a yellow duck named Bill and a pink donut named Bob and a red pepper named Mary. Put Mary on Bill.',

        'put red to the right',
        'put red to the left',
        'put yellow to the right',
        'put yellow to the left',
        'put duck to the left',
        'put duck to the right',
        'put pepper to the left',
        'put pepper to the right',
    ]



    # input_images=rp.random_batch(input_images, 8)
    # task_prompts=rp.random_batch(task_prompts, 3)

    def get_grid_image():
        image=rp.grid_concatenated_images(rows)
        return image

    rows=[]
    for task_prompt in rp.eta(task_prompts, title="task_prompt" ):
        rows.append([])
        for input_image in rp.eta(input_images, title="input_image"):
            output = run_llara(input_image, task_prompt)
            frame=output.frame

            import textwrap
            def wrap(text):
                return '\n'.join(textwrap.wrap(text,50))
            wrapped_task_prompt = wrap(task_prompt)
            frame=with_gridlines(frame)
            frame=rp.labeled_image(frame, wrapped_task_prompt, size_by_lines=True, size=30, font='Futura')
            frame=rp.bordered_image_solid_color(frame,'light red')
            rows[-1].append(frame)

            rp.display_image(get_grid_image())

    output_path = rp.get_absolute_path(rp.save_image(get_grid_image()))
    rp.open_file_with_default_application(output_path)
    rp.fansi_print("SAVED FILE: "+rp.fansi_highlight_path(output_path), "green", "bold")
    
    # rp.display_image(get_grid_image(),1)


    return get_grid_image()

if __name__ == '__main__':
    demo()
    # fire.Fire()
    # output=main('/Users/ryan/Downloads/12_last.png','Put the yellow corn on the brown thing')
    # rp.display_image(output.frame,1)
