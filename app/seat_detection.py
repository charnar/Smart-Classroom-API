import cv2
import numpy as np
from app.utills import linear_regression

class Seat:
    def __init__(self, x, y, width, height, area):
        self.position = (x, y)
        self.width = width
        self.height = height
        self.center = (int(x + width / 2), int(y + height / 2))
        self.area = area
        self.row = -1
        self.col = -1
        self.status = 0
        self.distance = -1


def display_seats(frame, seat_arr):
    for seat in seat_arr:
        cv2.rectangle(frame, (seat.position[0], seat.position[1]), (seat.position[0] + seat.width, seat.position[1] + seat.height), (0, 255, 0), 2)  # for testing purposes and display on the image
        cv2.putText(frame, str((seat.row, seat.col)), (seat.position[0], seat.position[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


 # A function to initialize the seating of the classroom
def arrange_seats(arr):   
    returned_arr = []
    n = 1   
    while (len(arr) != 0):
        temp_arr = []
        max_seat = max(arr, key=lambda s: s.position[1])    
        arr.remove(max_seat)                               
        temp_arr.append(max_seat)                          
        
        for selected_seat in arr:   
            if ((selected_seat.center[1] >= max_seat.position[1] - (0.4 * max_seat.height)) and (selected_seat.center[1] <= max_seat.position[1] + (0.9 * max_seat.height))):
                temp_arr.append(selected_seat)
            
            
        arr = [i for i in arr if i not in temp_arr] 

        temp_arr = sorted(temp_arr, key = lambda s: s.position[0], reverse = True)  
        
        m = 1
        for seat in temp_arr:   
            seat.row = n        
            seat.col = m       
            m = m + 1         

        returned_arr += temp_arr    
        n = n + 1                 

    col_lines = get_regression_lines(returned_arr) 
    row_lines = get_regression_lines(returned_arr, True) 

    return returned_arr  


def get_regression_lines(seats, horizontal=False):
    select_col = 1
    select_row = 1

    col_lines = []
    row_lines = []

    while True:
        co_x = []
        co_y = []
        for seat in seats:
            if select_col == seat.col and (not horizontal):
                co_x.append(seat.center[0])
                co_y.append(seat.center[1])

            elif select_row == seat.row and (horizontal):
                co_x.append(seat.center[0])
                co_y.append(seat.center[1])
                    
        if len(co_x) == 0:
            break

        beta0, beta1, line = linear_regression(co_y, co_x)
        if not horizontal:
            col_lines.append((co_x, co_y, beta0, beta1))

        else:
            row_lines.append((co_x, co_y, beta0, beta1))


        co_x = np.array(co_x) 
        co_y = np.array(co_y)

        if (not horizontal):
            select_col += 1
        else:
            select_row += 1

    if not horizontal:
        return col_lines

    else:
        return row_lines



def get_seat_contour(frame, contours): 
    seat_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if (area >= 200 and w >= 100 and y < 1000):
            seat_contours.append(Seat(x, y, w, h, area))
    
    return seat_contours


def get_seat_matrix(frame):
    (height, width) = frame.shape[:2]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame, (0, 0, 197), (179, 255, 255))

    # Return all white contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seats = get_seat_contour(frame, contours)
    
    return arrange_seats(seats)


    