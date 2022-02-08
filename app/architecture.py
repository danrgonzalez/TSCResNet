config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512,5,1), #my conv block goal is to achieve down to 18 from 22
    "P",
    # (512, 3, 2),
    # ["B", 8],
    # (1024, 3, 2),
    # ["B", 4],
  ]  # To this point is Darknet-53