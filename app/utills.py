import datetime
import calendar
import numpy as np
import cv2
import base64

# Returns the current time in UTC
def get_utc_time():
    date = datetime.datetime.utcnow()
    utc_time = calendar.timegm(date.utctimetuple())

    return utc_time


# Function calculates distance between all pairs and calculates closeness ratio.
def get_distances(boxes1, bottom_points, distance_w, distance_h):
    distance_mat = []
    bxs = []
    
    for i in range(len(bottom_points)):
        for j in range(len(bottom_points)):
            if i != j:
                dist = cal_dis(bottom_points[i], bottom_points[j], distance_w, distance_h)
          
                if dist <= 150:
                    closeness = 0
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness, dist])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                elif dist > 150 and dist <=180:
                    closeness = 1
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness, dist])
                    bxs.append([boxes1[i], boxes1[j], closeness])       
                else:
                    closeness = 2
                    distance_mat.append([bottom_points[i], bottom_points[j], closeness, dist])
                    bxs.append([boxes1[i], boxes1[j], closeness])
                
    return distance_mat, bxs


def get_transformed_seat_points(boxes, perspective_transform):
    for seat in boxes:
            x, y = seat.center
            pnts = np.array([[[int(x), int(y)]]] , dtype="float32")
            t_pnts = pnts[0][0]
            seat.centerTransformed = [int(t_pnts[0]), int(t_pnts[1])]


def linear_regression(x, y):
    N = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    B1_num = np.sum((x - x_mean) * (y - y_mean))
    B1_den = np.sum((x - x_mean)**2)
    B1 = B1_num / B1_den
    B0 = y_mean - (B1 * x_mean)
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
    
    return B0, B1, reg_line


def cal_dis(p1, p2, distance_w, distance_h):
    h = abs(p2[1] - p1[1])
    w = abs(p2[0] - p1[0])
    dis_w = float((w / distance_w) * 180)
    dis_h = float((h / distance_h) * 180)
    
    return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))


def get_physical_position(height, width, cal_scale_h, cal_scale_w, scale_h, scale_w):
    phy_h = round(int(height  * scale_h) / cal_scale_h, 2)
    phy_w = round(int(width * scale_w) / cal_scale_w, 2)
    return (phy_h, phy_w)


def get_window_position(height, width, scale_h, scale_w):
    bird_h = int(height * scale_h)
    bird_w = int(width * scale_w)

    return (bird_h, bird_w)


# Applies perspective transform to image
def perspective_transform_frame(frame, points):
    rows, cols, ch = frame.shape

    pts1 = np.float32([points[3], points[2], points[0], points[1]])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)

    transformed_image = cv2.warpPerspective(frame , M, (cols, rows))
    return transformed_image


def get_distance_from_seat(person_point, seat_point):
    p1 = np.array(person_point)
    p2 = np.array(seat_point)
    temp = p1 - p2

    return np.sqrt(np.dot(temp.T, temp))



def find_nearest_seat(person_point, seat_mat):
    temp_dists = []
    for seat in seat_mat:
        temp_dists.append(get_distance_from_seat(person_point, seat.center))
    
    idx = temp_dists.index(min(temp_dists))
    seat_mat[idx].status = 1
    
    return seat_mat[idx].center


# Sitting and Standing Threshold
def classify_box(width, height):
    return height / width >= 1.5


# Function to calculate bottom center for all bounding boxes and transform prespective for all points.
def get_transformed_points(boxes, pers_transform, seat_mat):
    bottom_points = []
    for box in boxes:
        standing_box = classify_box(box[2], box[3]) # width and height to see if they are sitting or standing
        pnts = np.array([[[int(box[0]+(box[2]*0.5)),int(box[1]+(box[3]*0.75))]]] , dtype="float32")
        bd_pnt = cv2.perspectiveTransform(pnts, pers_transform)[0][0]
        pnt = [int(bd_pnt[0]), int(bd_pnt[1])]

        if not standing_box:
            new_pnt = find_nearest_seat(pnt, seat_mat)
            pnt = [int(new_pnt[0]), int(new_pnt[1])]

        bottom_points.append(pnt)
        
    return bottom_points


def get_scale(W, H, bw, bh):
    dis_w = bw * 1.10
    dis_h = bh * 2.5
    
    return float(dis_w/W),float(dis_h/H)


# Function gives count for humans at high risk, low risk and no risk    
def get_count(distances_mat):
    r = []
    g = []
    y = []
    
    for i in range(len(distances_mat)):
        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])
                
    for i in range(len(distances_mat)):
        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])
        
    for i in range(len(distances_mat)):
        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])
   
    return (len(r),len(y),len(g))


def get_bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count, seat_mat, room_dim):

    h = frame.shape[0]
    w = frame.shape[1]
    
    win_h = int(h * scale_h)
    win_w = int(w * scale_w)

    room_h = room_dim[0]
    room_w = room_dim[1] 

    cal_scale_h = win_h / room_h 
    cal_scale_w = win_w / room_w
    #####################################


    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    white = (200, 200, 200)
    black = (0, 0, 0)


    blank_image = np.zeros((int(h * scale_h), int(w * scale_w), 3), np.uint8) 

    blank_image[:] = white
    warped_pts = []
    r = []
    g = []
    y = []

    persons = []
    high_risk_pair = []
    low_risk_pair = []
    safe_pair = []

    blank_image = cv2.flip(blank_image, 1)

    blank_image = cv2.circle(blank_image, (int(0* scale_h),int(0 * scale_w)), 5, red, 10)
    blank_image = cv2.putText(blank_image, "origin (0,0)",  (int(0 * scale_h)+30,int(0 * scale_w)+20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    

    blank_image = cv2.flip(blank_image, 1)
    blank_image = cv2.rotate(blank_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    ######################################
    #Calculate the ratio of the frame pixel and room  physical dimension
    #(Assume that the person coordinate from the provided code is correct)
    
    #Room dimension(h is the wider side of the room)
    room_h = room_dim[1]
    room_w = room_dim[0]

    cal_scale_h = win_h/room_h 
    cal_scale_w = win_w/room_w
    #######################################

    for s in seat_mat:
        center_x, center_y = s.centerTransformed
        phy_h, phy_w = get_physical_position(center_y, center_x, cal_scale_h, cal_scale_w, scale_h, scale_w)
        temp = "({:.2f},{:.2f})".format(phy_h, phy_w)
        s.physical_coords = (phy_h, phy_w)
        blank_image = cv2.circle(blank_image, (int(center_y * scale_h), int(center_x * scale_w)), 5, black, 10)
    
    
    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
                high_risk_pair.append({ 'person1' : (int(distances_mat[i][0][1] * scale_h), int(distances_mat[i][0][0] * scale_w)), 'person2' :(int(distances_mat[i][1][1] * scale_h), int(distances_mat[i][1][0] * scale_w)), 'distance' : distances_mat[i][3] / 100})

            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][1] * scale_h),int(distances_mat[i][0][0] * scale_w)), (int(distances_mat[i][1][1]* scale_h),int(distances_mat[i][1][0] * scale_w)), red, 2)
            
    for i in range(len(distances_mat)):
                
        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
                low_risk_pair.append({ 'person1' : (int(distances_mat[i][0][1] * scale_h), int(distances_mat[i][0][0] * scale_w)), 'person2' :(int(distances_mat[i][1][1] * scale_h), int(distances_mat[i][1][0] * scale_w)), 'distance' : distances_mat[i][3] / 100})
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])
        
            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][1] * scale_h),int(distances_mat[i][0][0] * scale_w)), (int(distances_mat[i][1][1]* scale_h),int(distances_mat[i][1][0] * scale_w)), yellow, 2)
            
    for i in range(len(distances_mat)):
        
        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
                safe_pair.append({ 'person1' : (int(distances_mat[i][0][1] * scale_h), int(distances_mat[i][0][0] * scale_w)), 'person2' :(int(distances_mat[i][1][1] * scale_h), int(distances_mat[i][1][0] * scale_w)), 'distance' : distances_mat[i][3] / 100})
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])
    
    #Fixed Physical Coordinate#
    for index, i in enumerate(bottom_points):
        blank_image = cv2.circle(blank_image, (int(i[1] * scale_h),int(i[0]  * scale_w)), 5, green, 10)
        phy_coordinates = get_physical_position(i[1], i[0], cal_scale_h, cal_scale_w, scale_h, scale_w)
        phy_h, phy_w = phy_coordinates

        bird_coordinates = get_window_position(i[1], i[0], scale_h, scale_w)
        bird_h, bird_w = bird_coordinates


        persons.append({'bird-eye-points' : bird_coordinates, 'physical-points' : phy_coordinates})
        
    for i in y:
        blank_image = cv2.circle(blank_image, (int(i[1] * scale_h),int(i[0]  * scale_w)), 5, yellow, 10)


    for i in r:
        blank_image = cv2.circle(blank_image, (int(i[1] * scale_h),int(i[0]  * scale_w)), 5, red, 10)
    
    
    blank_image = cv2.rotate(blank_image, cv2.ROTATE_90_CLOCKWISE)
    blank_image = cv2.flip(blank_image, 1)
    
    return blank_image, persons, (safe_pair, low_risk_pair, high_risk_pair)



def social_distancing_view(frame, distances_mat, persons, risk_count):
    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    pink = (255, 16, 240)
    
    for i in range(len(persons)):
        boxes, hand_boxes = persons[i]

        x,y,w,h = boxes
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),green,2)

        for hand in hand_boxes:
            x1, y1, w1, h1 = hand
            center_x1 = x + int(x1 + (w1 / 2))
            center_y1 = y + int(y1 + (h1 / 2))

            top_left_x1 = x + x1
            top_left_y1 = y + y1

            x1_length = top_left_x1 + w1
            y1_length = top_left_y1 + h1

            frame = cv2.rectangle(frame,(top_left_x1, top_left_y1),(x1_length, y1_length),pink,2)


    for i in range(len(distances_mat)):
        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        
        if closeness == 1:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),yellow,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),yellow,2)
            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),yellow, 2) 
            
    for i in range(len(distances_mat)):
        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        
        if closeness == 0:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),red,2)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),red,2)
            
            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 2)
            
            
    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    cv2.putText(pad, "-- HIGH RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(pad, "-- LOW RISK : " + str(risk_count[1]) + " people", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    frame = np.vstack((frame,pad))
            
    return frame


def get_image_from_base64(base64_string):
    image_bytes = base64.b64decode(base64_string)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def get_base64_from_cv2(buffer):
    return base64.b64encode(buffer)


def tuple_string_to_float_array(tuple_string):
    tuple_string = tuple_string.replace("(", "")
    tuple_string = tuple_string.replace(")", "")
    tuple_string = tuple_string.replace(",", "")
    return list(map(float, tuple_string.split()))


def tuple_string_to_int_array(tuple_string):
    tuple_string = tuple_string.replace("(", "")
    tuple_string = tuple_string.replace(")", "")
    tuple_string = tuple_string.replace(",", "")
    return list(map(int, tuple_string.split()))
