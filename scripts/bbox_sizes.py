import cv2

def find_small_bounding_boxes(data):
    for img in data.keys():
        for j in data[img]["ground_truth"]:
            x1 = int(j[0])
            y1 = int(j[1])
            x2 = int(j[2])
            y2 = int(j[3])

            if x1 == x2 or y1 == y2:
                print(f"{img} has a 0x0 bounding box")
                continue

            if x2 - x1 < 10 or y2 - y1 < 10:
                print(f"{img} has a small bounding box")

def find_large_bounding_boxes(data):
    for img in data.keys():
        width, height = cv2.imread(img).shape[:2]

        for j in data[img]["ground_truth"]:
            x1 = int(j[0])
            y1 = int(j[1])
            x2 = int(j[2])
            y2 = int(j[3])
            
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                print(f"{img} has a large bounding box")
