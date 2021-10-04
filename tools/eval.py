import argparse

import csv
import cv2
import numpy as np

import detect_ymeiji


def mask_mouse(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 緑色のHSVの値域1
    # hsv_min = np.array([25, 25, 30])
    # hsv_max = np.array([50, 50, 60])

    # hsv_min = np.array([140, 10, 30])
    # hsv_max = np.array([170, 45, 50])

    hsv_min = np.array([140, 10, 30])
    hsv_max = np.array([175, 50, 50])

    mask = cv2.inRange(img, hsv_min, hsv_max)

    white = np.full(img.shape, 255, dtype=img.dtype)
    background = cv2.bitwise_and(
        white, white, mask=mask
    )  # detected mouse area becomes white

    inv_mask = cv2.bitwise_not(mask)  # make mask for not-mouse area
    extracted = cv2.bitwise_and(img, img, mask=inv_mask)

    masked = cv2.add(extracted, background)

    return masked


def detect_arm(img, coefs_line):
    arms_name = ["A", "B", "C"]
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    cv2.circle(
        img,
        (int(coefs_line[3][0]), int(coefs_line[3][1])),
        int(coefs_line[4][0]),
        (0, 0, 255),
        2,
    )

    # img_maskgreen = mask_green(img)
    # img_gray = cv2.cvtColor(img_maskgreen, cv2.COLOR_BGR2GRAY)
    # _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # _, img_binary1 = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    img_maskmouse = mask_mouse(img)
    img_gray = cv2.cvtColor(img_maskmouse, cv2.COLOR_BGR2GRAY)
    _, img_binary2 = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)
    # img_er = cv2.erode(img_binary2, kernel, iterations=1)
    # img_dl = cv2.dilate(img_er, kernel, iterations=3)
    # img_opening2 = cv2.morphologyEx(img_binary2, cv2.MORPH_OPEN, kernel)
    img_closing = cv2.morphologyEx(img_binary2, cv2.MORPH_CLOSE, kernel)

    # img = img_closing
    # img2 = cv2.bitwise_not(img_closing)
    contours, hierarchy = cv2.findContours(
        img_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    arm = ""
    if contours:
        contours = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contours)
        center_x = x + w / 2
        center_y = y + h / 2
        d_min = 10000

        r = (
            (center_x - coefs_line[3][0]) ** 2 + (center_y - coefs_line[3][1]) ** 2
        ) ** (1 / 2)
        # print(r)
        # print("min", coefs_line[4][0])
        # print("max", coefs_line[4][1])
        if r < coefs_line[4][0] or r > coefs_line[4][1]:
            pass
        else:
            for i in range(3):
                d = (
                    abs(
                        coefs_line[i][0] * center_x
                        + coefs_line[i][1] * center_y
                        + coefs_line[i][2]
                    )
                ) / ((coefs_line[i][0] ** 2 + coefs_line[i][1] ** 2) ** (1 / 2))
                if d < d_min:
                    d_min = d
                    arm = arms_name[i]

            # print(f"\r{w*h}\t{w * h>500}\t{arm}", end="")

            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return img, arm


def get_coefs(points):
    coefs = np.zeros((5, 3))
    coefs[3][0] = points[0][0]  # center_x
    coefs[3][1] = points[0][1]  # center_y
    # coefs[4][0]: r_min
    # coefs[4][1]: r_max
    coefs[4][1] = 99999

    for i in range(3):
        m = (points[i + 1][1] - points[0][1]) / (points[i + 1][0] - points[0][0])
        a = m
        b = -1
        c = points[0][1] - m * points[0][0]
        coefs[i] = [a, b, c]
        r = (
            (points[i + 1][1] - points[0][1]) ** 2
            + (points[i + 1][0] - points[0][0]) ** 2
        ) ** (1 / 2)
        if r < coefs[4][1]:
            coefs[4][1] = r
    coefs[4][0] = coefs[4][1] * 0.15
    # print(coefs)

    return coefs


def create_armslist(file, coefs, start, end, show=False):
    arms = []
    prev_arm = ""

    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print(f"Can't open {file}")
        exit()
    """
    ret, frame = cap.read()

    if shift:
        cap.set(0, shift)
        ret, frame = cap.read()
    """
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * cap.get(cv2.CAP_PROP_FPS)))
    try:
        while True:
            ret, frame = cap.read()
            # print(ret, frame)
            if ret:
                frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                now_second = int(frame_no / cap.get(cv2.CAP_PROP_FPS))
                min, sec = divmod(now_second, 60)
                # print(frame_no, now_second)
                if frame_no > end * cap.get(cv2.CAP_PROP_FPS):
                    break
                w, h, _ = frame.shape
                frame, arm = detect_arm(frame, coefs)
                if prev_arm != arm and arm != "":
                    arms.append(arm)
                    prev_arm = arm
                    print(arm)

                # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
                if show:
                    # time
                    cv2.putText(
                        frame,
                        f"{min}:{sec:02}",
                        org=(int(w * 0.05), int(h * 0.05)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=3,
                        lineType=cv2.LINE_4,
                    )
                    cv2.putText(
                        frame,
                        f"{min}:{sec:02}",
                        org=(int(w * 0.05), int(h * 0.05)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_4,
                    )
                    # arm_name
                    cv2.putText(
                        frame,
                        prev_arm,
                        org=(int(w * 0.1), int(h * 0.1)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=3.0,
                        color=(255, 255, 255),
                        thickness=10,
                        lineType=cv2.LINE_4,
                    )
                    cv2.putText(
                        frame,
                        prev_arm,
                        org=(int(w * 0.1), int(h * 0.1)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=3.0,
                        color=(0, 0, 0),
                        thickness=4,
                        lineType=cv2.LINE_4,
                    )
                    cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("中断されました")
                    exit()
            else:
                break
    except KeyboardInterrupt:
        print("中断されました")
        exit()

    cap.release()
    cv2.destroyAllWindows()

    return arms


def check_filelen(file, shift):
    eightmin = 8 * 60  # second
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print(f"Can't open {file}")
        exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start = shift * fps
    end = (shift + eightmin) * fps
    try:
        assert start < count and end < count, "動画が8分未満になるため評価を開始できません"
    except AssertionError as err:
        print("AssertionError :", err)
    cap.release()
    return shift, shift + eightmin


def has_duplicates(seq):
    return len(seq) != len(set(seq))


def evaluation(arms):
    activity = len(arms) - 1
    working_memory = 0
    for i in range(len(arms) - 2):
        # print(i, arms[i : i + 3])
        if not has_duplicates(arms[i : i + 3]):
            working_memory += 1
    working_memory = working_memory * 100 / (activity - 1)

    return activity, working_memory


def save(file, arms, activity, working_memory):
    activity = str(activity)
    working_memory = str(working_memory)
    header = ["route", "activity", "working_memory"]
    contents = [arms, activity, working_memory]

    if "csv" in file:
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            contents = [arms[0], activity, working_memory]
            writer.writerow(contents)
            for i in range(1, len(arms)):
                contents = [arms[i], "", ""]
                writer.writerow(contents)

    elif "txt" in file:
        contents = ["".join(arms), activity, working_memory]
        with open(file.replace("csv", "txt"), "w") as f:
            writer = csv.writer(f)
            for i in range(len(header)):
                f.write(f"{header[i]}:\n{contents[i]}\n\n")

    print(f"Saved result to {file}")


def main(args):
    file = args.file
    show = args.show
    shift = args.shift

    outfile = file
    ext = outfile.rsplit(".", 1)[-1]
    outfile = outfile.replace(ext, args.output).replace("data", "result")

    start, end = check_filelen(file, shift)  # second

    points = np.zeros((4, 2), np.int)
    # point_name = ["Center", "A", "B", "C"]
    center, edges = detect_ymeiji.getpoints(file)
    points[0] = center
    for i in range(3):
        points[i + 1] = edges[i]
    # points = np.array([[485, 301], [250, 376], [533, 73], [669, 465]])
    coefs = get_coefs(points)

    if show:
        print("中断するにはqを押してください")
    else:
        print("中断するにはCtrl+cを押してください")
    arms = create_armslist(file, coefs, start, end, show)
    activity, working_memory = evaluation(arms)
    print("activity:", activity)
    print("working_memory:", f"{working_memory:.2f}")
    save(outfile, arms, activity, working_memory)


def debug1(args):
    img = cv2.imread("mouse.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("frame", img)
    # img = np.where(img > 100, 0, img)
    img_max = np.max(img, axis=0)
    img_max = np.max(img_max, axis=0)
    print(img_max)
    img_min = np.min(img, axis=0)
    img_min = np.min(img_min, axis=0)
    print(img_min)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def debug2(args):
    arms = ["A", "F", "D"]
    activity = 27
    working_memory = 65.635
    save("./result/01504.txt", arms, activity, working_memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation for mouse Y_meiji experiment."
    )
    parser.add_argument(
        "file",
        type=str,
        help="video file path (ex: ./data/01504.MTS)",
    )
    parser.add_argument(
        "-o",
        "--output",
        choices=["csv", "txt"],
        default="csv",
        help="format of output file",
    )
    parser.add_argument(
        "-s",
        "--shift",
        type=float,
        default=0,
        help="video shift",
    )
    parser.add_argument("--show", action="store_true", help="show visualization")
    args = parser.parse_args()
    # debug(args)
    # debug2(args)
    main(args)
