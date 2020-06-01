import os
import cv2


video_folder_path = "videos"    # Folders to load videos and store frames
data_folder_path = "data"

if not os.path.isdir(data_folder_path): # Creates data folder if there is none
    os.mkdir(data_folder_path)

os.chdir(data_folder_path)

vid = 0
for file in os.listdir("../" + video_folder_path):  # Loops through videos
    if file[0] == ".":
        continue

    file_path = "../" + video_folder_path + "/" + file

    cap = cv2.VideoCapture(file_path)

    i = 0

    while(cap.isOpened()):          # Saves every 100th frame in data folder
        ret, frame = cap.read()
        if ret == False:
            break
        if i%100==0:
            cv2.imwrite('video' + str(vid) + 'img'+str(i)+'.jpg',frame)
            print(f"Img number {i} written.")
        i+=1
    vid += 1

    cap.release()
    cv2.destroyAllWindows()
