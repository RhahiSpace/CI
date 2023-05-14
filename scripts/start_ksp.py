# %%
import logging
import os
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
from Xlib import X, display
from Xlib.ext.xtest import fake_input


class TemplateNotFoundError(Exception):
    pass

class KerbalNotFoundError(Exception):
    pass

class WindowLaunchFailedError(Exception):
    pass


WIN_Y_OFFSET = 25
KSP_WIDTH = 1280
KSP_HEIGHT = 720


def start_screen():
    if os.environ.get('DISPLAY') is None:
        os.environ['DISPLAY'] = ':0'
    # if os.environ.get('DISPLAY') is None:
    #     # if display was not given, start a virtual one.
    #     xvfb = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1440x900x24'], start_new_session=True)
    #     os.environ['DISPLAY'] = ':99'
    #     logging.info(f'Started Xvfb with PID {xvfb.pid}')
    #     time.sleep(1)
    #     if xvfb.returncode:
    #         raise WindowLaunchFailedError
    #     return xvfb


def start_ksp():
    # /home/rhahi/.steam/steam/steamapps/common/Kerbal Space Program/KSP.x86_64
    KERBAL_PATH = os.environ.get('KERBAL_PATH')
    if KERBAL_PATH:
        kerbal = subprocess.Popen(KERBAL_PATH, start_new_session=True)
        logging.info(f'Started KSP with PID {kerbal.pid}')
        time.sleep(1)
        return kerbal
    else:
        raise KerbalNotFoundError("KSP not found, please specify KERBAL_PATH environment variable.")


def take_screenshot(geometry, name=""):
    raw_image = root.get_image(x=0, y=0, width=geometry.width, height=geometry.height, format=X.ZPixmap, plane_mask=0xFFFFFFFF)
    image = Image.frombytes('RGB', (geometry.width, geometry.height), raw_image.data, 'raw', 'BGRX')
    if name:
        image.save(f'./run/{name}.png')
    return image


def take_raw_screenshot(root, geometry):
    raw_image = root.get_image(x=0, y=0, width=geometry.width, height=geometry.height, format=X.ZPixmap, plane_mask=0xFFFFFFFF)
    image = np.frombuffer(raw_image.data, dtype=np.uint8).reshape(geometry.height, geometry.width, -1)
    return image


def match_template(image, template, threshold=0.9):
    screenshot = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    template = cv2.imread(template)
    _, w, h = template.shape[::-1]
    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    center = None
    for pt in zip(*loc[::-1]):
        cv2.rectangle(screenshot, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        center = (pt[0] + w // 2, pt[1] + h // 2)
    cv2.imwrite('./run/match.png', screenshot)
    return center


def click_location(dsp: display.Display, location):
    fake_input(dsp, X.MotionNotify, x=location[0], y=location[1])
    dsp.sync()
    fake_input(dsp, X.ButtonPress, 1)
    dsp.sync()
    fake_input(dsp, X.ButtonRelease, 1)
    dsp.sync()


def wait_until_stable(
        root,
        geometry,
        threshold=3,
        timeout=20,
        roi=None):
    logging.debug('waiting for stable image')
    prev = None
    start = time.time()
    roi_start = [geometry.width//2 - KSP_WIDTH //2, geometry.height//2 - KSP_HEIGHT //2 + WIN_Y_OFFSET]
    if roi:
        roi_end = [roi_start[i] + roi[1][i] for i in range(2)]
        roi_start = [roi_start[i] + roi[0][i] for i in range(2)]
    else:
        roi_end = [roi_start[0] + KSP_WIDTH, roi_start[1] + KSP_HEIGHT]

    debug_image = take_screenshot(geometry)
    screenshot = cv2.cvtColor(np.array(debug_image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(screenshot, roi_start, roi_end, (0, 255, 0), 2)
    cv2.imwrite('./run/roi.png', screenshot)

    while time.time() - timeout < start:
        image = take_raw_screenshot(root, geometry)
        roi = image[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]

        if prev is not None:
            diff = cv2.absdiff(roi, prev)
            mean_diff = np.mean(diff)
            if mean_diff < threshold:
                logging.debug('image is stable')
                return True
            logging.debug('image is not stable')
        prev = roi
        time.sleep(0.5)
    logging.warning('Timeout occured while waiting for stable image.')
    return False


def find_template(template, image=None, timeout=20, interval=1, **kwargs):
    logging.debug('looking for a matching template')
    start = time.time()
    while time.time() - timeout < start:
        image = take_screenshot(geometry, name="screenshot-launch")
        button = match_template(image, template, **kwargs)
        time.sleep(interval)
        if button is None:
            continue
        return button
    logging.warning('Timeout occured while waiting for stable image.')
    raise TemplateNotFoundError


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s %(message)s')
    xvfb = None
    kerbal = None
    try:
        xvfb = start_screen()
        kerbal = start_ksp()
        dsp = display.Display()
        root = dsp.screen().root
        geometry = root.get_geometry()

        logging.info('waiting for main menu to appear')
        take_screenshot(geometry, name="screenshot1-launch")
        find_template('./template/menu.png', timeout=300, interval=5)
        wait_until_stable(root, geometry, threshold=3, timeout=20, roi=((0, 0), (int(KSP_WIDTH//2.2), KSP_HEIGHT)))

        logging.info('clicking start game button')
        take_screenshot(geometry, name="screenshot2-start")
        button = find_template('./template/start.png')
        click_location(dsp, button)
        wait_until_stable(root, geometry, threshold=3, timeout=20)

        logging.info('clicking resume button')
        take_screenshot(geometry, name="screenshot3-resume")
        button = find_template('./template/resume.png')
        click_location(dsp, button)
        wait_until_stable(root, geometry, threshold=3, timeout=20)

        logging.info('loading save game "ci"')
        take_screenshot(geometry, name="screenshot4-load")
        button = find_template('./template/load.png')
        click_location(dsp, button)

        image = take_screenshot(geometry, name="screenshot5-end")

        logging.info('done')
    except KerbalNotFoundError:
        pass
    except (Exception, KeyboardInterrupt, SystemExit):
        kerbal.terminate()
    finally:
        if kerbal:
            print(kerbal.pid)
        else:
            print(0)
