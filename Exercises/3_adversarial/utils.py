

def show_points(points):
    board = np.zeros((HEIGHT, WIDTH))
    for p in points:
        board[p] = 1
    print(board)
