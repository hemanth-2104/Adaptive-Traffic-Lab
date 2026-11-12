from pathlib import Path
import re
from PIL import Image,ImageDraw,ImageFont
from .data import TAG,CAMERAS

DIRECTION_CAM=[('back left','CAM_BACK_LEFT'),('back right','CAM_BACK_RIGHT'),('front left','CAM_FRONT_LEFT'),('front right','CAM_FRONT_RIGHT'),('back','CAM_BACK'),('front','CAM_FRONT')]

def cameras_for_question(question):
    refs=TAG.findall(question)
    if refs:return list(dict.fromkeys(t[1] for t in refs))
    for phrase,cam in DIRECTION_CAM:
        if phrase in question.lower():return [cam]
    return ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']


def compose_view(images,image_root,questions,all_cameras=False,width=448,height=252):
    cameras=CAMERAS if all_cameras else list(dict.fromkeys(c for q in questions for c in cameras_for_question(q)))
    cameras=[c for c in cameras if c in images]
    if not cameras:raise ValueError('No usable camera views')
    cols=min(3,len(cameras));rows=(len(cameras)+cols-1)//cols;bar=26
    panel=Image.new('RGB',(width*cols,(height+bar)*rows),(246,246,244));draw=ImageDraw.Draw(panel)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    except OSError:
        try:font=ImageFont.truetype('DejaVuSans.ttf',16)
        except OSError:font=ImageFont.load_default(size=16)
    tags={t for q in questions for t in TAG.findall(q)}
    for i,cam in enumerate(cameras):
        path=(Path(image_root)/images[cam]).resolve();root=Path(image_root).resolve()
        if not path.is_relative_to(root):raise ValueError('Image escaped dataset root')
        with Image.open(path) as raw:
            ow,oh=raw.size;im=raw.convert('RGB').resize((width,height))
        x=(i%cols)*width;y=(i//cols)*(height+bar)
        panel.paste(im,(x,y+bar));draw.text((x+8,y+3),cam,font=font,fill='black')
        for cid,cc,px,py in tags:
            if cc!=cam:continue
            tx=x+float(px)/ow*width;ty=y+bar+float(py)/oh*height
            draw.ellipse((tx-5,ty-5,tx+5,ty+5),fill='red',outline='white',width=1)
            b=draw.textbbox((tx+6,ty-20),cid,font=font);draw.rectangle(b,fill='white');draw.text((tx+6,ty-20),cid,font=font,fill='red')
    return panel
