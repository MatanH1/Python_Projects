######
### Name : Matan Hirschhorn, User name : matanh10, ID: 207682949
import math
import copy

WHITE = 0
BLACK = 1
NEUTRAL = -1
def zero_row(row, count):
    """
    takes a row, makes it all white till the end.
    :param row: row row we'd like to whiten
    :param count: number of blacks we will whiten.
    :return: Nothing, the row just becomes white.
    """
    for i in range(count):
        row[i] = WHITE


def get_available_cells(row):
    """
    Checks the number of cells until the next white cell.
    :param row: Row we will check
    :return: Number of cells we passed until the next white one
    """
    count = 0
    for cell in row:
        if cell == WHITE:
            break
        count += 1

    return count


def block_fits_in_row(row,block_size):
    """
    Checks if a block can fit inside the row.
    :param row: Row we will check on
    :param block_size: Size of block we are checking
    :return: Boolean value if it can possibly fit or not.
    """
    available_cells = get_available_cells(row)
    if available_cells < block_size:
        return False
    if available_cells > block_size and row[block_size] == BLACK:
        return False
    return True


def get_row_variations_help(row, blocks):
    """

    :param row: Row we have, with black, white and neutral cells.
    :param blocks: Values of blocks we will fill
    :return: All rows that can come from those values.
    """
    # We finished the blocks and therefore we can return a possible row now
    if len(blocks) == 0:
        zero_row(row, len(row))
        return [row]

    # We got to the end of the row AND there are still blocks to be positioned, therefore this is not a possible row
    if len(row) < 1:
        return []

    # We check that all blocks can fit in row
    if sum(blocks) + len(blocks) -1 > len(row):
        return []

    # Now we will try to position the first block in blocks list in our row
    # We will call the first block, block_size
    block_size = blocks[0]
    possible_rows = []
    if block_fits_in_row(row, block_size):
        if len(row) == block_size:
            new_block = [BLACK]*block_size
        else:
            new_block = [BLACK]*block_size + [WHITE]
        all_rows = get_row_variations(row[block_size+1:],blocks[1:])
        for next_possible_row in all_rows:
            possible_row = new_block + next_possible_row
            possible_rows.append(possible_row)

    if not row[0] == BLACK:
        all_rows = get_row_variations(row[1:], blocks)
        for next_possible_row in all_rows:
            possible_row = [WHITE]+next_possible_row
            possible_rows.append(possible_row)

    return possible_rows


def get_row_variations(row,blocks):
    """
    The function we've worked for until now with all helping functions.

    :param row: Row we will check.
    :param blocks: Values of blocks given
    :return: All rows possible from the values, using the helper function above

    """
    lst = []
    lst = get_row_variations_help(row,blocks)
    return lst

def get_intersection_row(rows):
    """

    :param rows: All possible rows from the variation function.
    :return: Intersction among them all.
    """
    if rows == []:
        return None
    if len(rows) == 1:
        return rows[0]
    changed_definetly = False
    lst = []
    for i in range(len(rows[0])):
        for j in range(1, len(rows)):
            if rows[0][i] != rows[j][i]:
                if rows[j][i] != NEUTRAL:
                    changed_definetly = True
                    break
        if changed_definetly:
            lst.append(NEUTRAL)
        else:
            lst.append(rows[j][i])
        changed_definetly = False
    return lst


def fill_up(row, constraints,i):
    """
    Mixes the 2 functions above. takes all variations of row, retunrs the intersections of them.
    :param row: Row we will egt the variations of
    :param constraints: All block limits of the row
    :param i: Index of constranit we will check.
    :return: Intersction of all row variations.
    """
    all_options = get_row_variations(row, constraints[i])
    if not all_options:
        return None
    new_row = get_intersection_row(all_options)
    return new_row


def board_concluding(board, constraints):
    """
    Board updates itself until we have no information to renrew in it.
    :param board: Board we will update.
    :param constraints: Constraints of blocks on each row.
    :return: New board, updated according to all information from different functions.
    """
    if not board:
        return None
    something_new = True  ## Do we need to go on another round?
    while something_new:
        something_new = False
        for i in range(len(board)):  ## running on rows
            potential_row = fill_up(board[i], constraints[0],i)
            if potential_row is None:
                return None
            if potential_row != board[i]:
                board[i] = potential_row
                something_new = True
        for i in range(len(board[0])):  ## running on columns:
            col = []
            for j in range(len(board)):
                col.append(board[j][i])
            potential_col = fill_up(col, constraints[1],i)
            if potential_col is None:
                return None
            if potential_col != col:
                for j in range(len(board)):
                    board[j][i] = potential_col[j]
                    something_new = True
    return board


def draw_board(constraints):
    """
    Draws a plain board
    :param constraints: Constrains of rows
    :return:  New Board.
    """
    board = []
    for i in range(len(constraints[0])):
        board.append([NEUTRAL] * len(constraints[1]))
    return board


def solve_easy_nonogram(constraints):
    """
   Solves the nonogram as much as it can.
    :param constraints: row constraints
    :return: Board, after being solved as far as it can get.
    """
    board = draw_board(constraints)
    return board_concluding(board, constraints)


def solve_nonogram_helper(board, constraints, row,lst):
    """
    Helper function, to eventually give all options to complete the board.
    :param board: Board, solved as far as it got in previous function.
    :param constraints: Constraints on rows and columns.
    :param row: Number of row we will check
    :param lst: Empty list, will eventually include all possible solutions.
    :return: List of all possible solutions to complete the game.
    """
    if row == len(board):
        lst.append(copy.deepcopy(board))
        return
    all_options = get_row_variations(board[row],constraints[0][row])
    saver = copy.deepcopy(board[row])
    new_board = copy.deepcopy(board)
    for i in all_options:
        board=copy.deepcopy(new_board)
        board[row] = i
        board = board_concluding(board,constraints)
        if board is None:
            continue
        if NEUTRAL not in board:
            lst.append(copy.deepcopy(board))
            continue

        #lst.append(potential_board)
        solve_nonogram_helper(board,constraints,row+1,lst)
    board = new_board



def solve_nonogram(constraints):
    """
    Gives all options to complete the board, using the previous helper function.
    :param constraints: Constraints on rows and columns.
    :return: All possible solved boards.
    """
    board = solve_easy_nonogram(constraints)
    if board is None:
        return []
    lst = []
    solve_nonogram_helper(board,constraints,0,lst)
    return lst


def count_row_variations(length, blocks):
        """
        Counts number of possible rows with the values given.
        :param length: Length of the row
        :param blocks: List of all blocks in the row.
        :return: Number of possible rows.8
        """
        sum_blocks = sum(blocks)
        if sum_blocks >= length:
            return 0
        n = length - sum_blocks+1
        k =  sum_blocks
        return int((math.factorial(n))/(math.factorial(k)*math.factorial(n-k)))