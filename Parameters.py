import cv2

# parameters of background model
BGSM_HISTORY = 25
BGSM_BACKGROUND_RATIO = 0.25

# parameters of blob detector
MIN_AREA_BLOB = 10
MIN_CIRCULARITY = 0.1
MIN_CONVEXITY = 0.25
MIN_INERTIA_RATIO = 0.1

# parameters for foreground optimization using morphological operations
KERNEL_SIZE = 5
KERNEL_CIRCLE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))

# maximum elapsed time between valid object detections in frames
TIME_BETWEEN_BLOBS = 50

# minimum object size for initial droplet detection in px
MIN_DROPLET_RADIUS = 5
# search radii for second sampling points on principal and rebound trajectories in px
MAX_RADIUS_SECOND_DROPLET_MAIN = 40
MAX_RADIUS_SECOND_DROPLET_REBOUND = 40
# distance added to search radius in px
ADD_TO_SEARCH_RADIUS = 30