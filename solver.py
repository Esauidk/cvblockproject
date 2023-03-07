from uwimg import *
red = 0.0051
yellow = 0.168
green = 0.333
blue = 0.667

def place_piece(color, piece):
    col = 0
    if (color == "yellow"):
        col = yellow
    elif (color == "blue"):
        col = blue
    elif (color == "green"):
        col = green
    elif (color == "red"):
        col = red
    im = load_image("data/blokus/counter1.png")
    square = center_board(im, 0.6)
    save_image(square, "squared_board")
    mini = minimize_board(square, 0.35)
    placements = copy_image(mini)

    find_piece_placement(mini, col, piece)
    save_image(mini, "next_move")

    rgb_to_hsv(placements)
    show_illegal_squares(placements, col)
    show_available_corners(placements, col)
    hsv_to_rgb(placements)
    save_image(placements, "decision_map")

place_piece("red", "I1")


# def create_mask():
#     im = load_image("data/blokus/counter1.png")
#     rgb_to_hsv(im)
#     sat_thresh = 0.5
#     sat_filter = make_sat_filter(6)

#     high_sat = convolve_image(im, sat_filter, 1)
#     set_image_above_sat(high_sat, 1, 1, sat_thresh)
#     set_image_above_sat(high_sat, 2, 1, sat_thresh)
#     set_image_below_sat(high_sat, 2, 0, sat_thresh)
#     hsv_to_rgb(high_sat)
#     hsv_to_rgb(im)
#     scan_for_corners(im, high_sat)

#     save_image(im, "corners")

def scan_corners():
    im = load_image("data/blokus/counter1.png")
    sat_thresh = 0.7
    dist = scan_for_corners(im, sat_thresh)
    save_image(im, "corners")
    rotated = rotate_45(im)
    rot_dist = scan_for_corners(rotated, sat_thresh)
    save_image(rotated, "rotated_corners")
    if (dist > rot_dist):
        print("using default")
    else:
        print("using rotated")

def rotate_image():
    im = load_image("data/blokus/counter1.png")
    rotated = rotate_45(im)
    save_image(rotated, "corners")

#scan_corners()
#rotate_image()