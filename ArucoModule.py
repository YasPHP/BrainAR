import cv2
import cv2.aruco as aruco
import numpy as np
import os


def loadAugImages(path):
    """
    Loads the images to be overlaid on the arUco markers via a dictionary.

    :param path: the directory folder path with the marker images and their ids
    :return: a dict with {id: augmented image}
    """
    markersList= os.listdir(path)

    # total number of overlaid markers available
    totalMarkers = len(markersList)

    # # prints the file names of the overlaid marker images in the folder path
    # print(markersList)

    # prints the total number of overlaid markers available
    print("Total # of markers:", totalMarkers)

    imgAugDict = {}

    for imgAugPath in markersList:
        key = str(os.path.splitext(imgAugPath)[0])
        print(key)
        imgAug = cv2.imread(f'{path}/{imgAugPath}')
        imgAugDict[key] = imgAug

    # print(imgAugDict)
    return imgAugDict



def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    """
    Returns the found ArUco markers in a frame with details.

    :param img: the image where the aruco markers exist
    :param markerSize: the size of the markers
    :param totalMarkers: the total number of markers (in chosen ArUco DICT version)
    :param draw: the bboxes drawn around the detected markers
    :return: the detected bboxes and aruco id of the detected markers
    """

    # converting image to gray
    imgGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # customized ArUco Dictionary key
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')

    # defining the ArUco Dictionary
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray,
                                               arucoDict,
                                               parameters=arucoParam)

    # prints ids of arUco markers detected
    # print(ids)

    # draws boundary boxes around detected arUco markers
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]


# CONTINUE: AT 17:57 (https://www.youtube.com/watch?v=v5a7pKSOJd8)


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    """
    Overlays the target image over top the detected marker.

    :param bbox: the boundary box of the aruco marker (four corner points)
    :param id: the id of the overlaid image to be displayed
    :param img: the image that will be drawn on top of
    :param imgAug: the displayed image over top the aruco marker
    :param drawId: the id displayed over top the aruco marker
    :return: the image frame with the augmented image overlaid
    """
    top_left = bbox[0][0][0], bbox[0][0][1]
    top_right = bbox[0][1][0], bbox[0][1][1]
    bottom_right = bbox[0][2][0], bbox[0][2][1]
    bottom_left = bbox[0][3][0], bbox[0][3][1]

    height, width, channels = imgAug.shape

    pts1 = np.array([top_left, top_right, bottom_right, bottom_left])
    pts2 = np.float32([[0,0], [width,0], [width, height], [0, height]])

    matrix, _ = cv2.findHomography(pts2, pts1)

    # warps image with augmented image, matrix, and image shape
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
    imgOut = img + imgOut

    # # draws the ArUco Marker IDs onto the Live Video window
    # if drawId:
    #     cv2.putText(imgOut, str(id), top_left,
    #                 cv2.FONT_HERSHEY_PLAIN, 2,
    #                 (255, 0, 255), 2)

    return imgOut


def main():
    # opens live video camera
    # NOTE: built-in camera is 0, multiple cameras are scaled to 1, 2, 3, etc.
    cap = cv2.VideoCapture(0)

    # the image to be overlaid on top of the aruco marker
    # imgAug = cv2.imread("MarkerOverlays/frontalLobe.jpg")

    imgAugDict = loadAugImages("MarkerOverlays")

    while True:
        # loop continuously reading frame-by-frame
        success, img = cap.read()
        if success:
            # find the aruco markers in the image frame (img/frame)
            arucoFound = findArucoMarkers(img)

            # loops through each found marker and augments it
            if len(arucoFound[0]) != 0:
                # loops through both at the same time
                for bbox, id in zip(arucoFound[0], arucoFound[1]):
                    img = augmentAruco(bbox, id, img, imgAug)

            # a frame was successfully read
            # show camera feed in a window
            cv2.imshow("Live Video", img)


            # check for a key pressed event and break the camera loop
            k = cv2.waitKey(5) & 0xFF

            # click the escape button on keyboard to exit camera view
            if k == 27:
                # closes the webcam window
                cv2.destroyAllWindows()
                cap.release()
                break

        # frame wasn't read, handle that problem:
        else:
            # closes the webcam window
            cv2.destroyAllWindows()
            cap.release()
            break


# main program executor
if __name__ == "__main__":
    main()