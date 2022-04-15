from typing import Union

from PIL import Image, ImageDraw


def get_cell_sizes(cell_size: Union[int, list, tuple]):
    """Handle multiple type options of `cell_size`.

    In order to keep the old API of following functions, as well as add
    support for non-square grids we need to check cell_size type and
    extend it appropriately.

    Args:
        cell_size: integer of tuple/list size of two with cell size 
            in horizontal and vertical direction.

    Returns:
        Horizontal and vertical cell size.
    """
    if isinstance(cell_size, int):
        cell_size_vertical = cell_size
        cell_size_horizontal = cell_size
    elif isinstance(cell_size, (tuple, list)) and len(cell_size) == 2:
        # Flipping coordinates, because first coordinates coresponds with height (=vertical direction)
        cell_size_vertical, cell_size_horizontal = cell_size
    else:
        raise TypeError("`cell_size` must be integer, tuple or list with length two.")

    return cell_size_horizontal, cell_size_vertical


def draw_grid(rows, cols, cell_size=50, fill='black', line_color='black'):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)

    width = cols * cell_size_x
    height = rows * cell_size_y
    image = Image.new(mode='RGB', size=(width, height), color=fill)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height

    for x in range(0, image.width, cell_size_x):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=line_color)

    x = image.width - 1
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=line_color)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, cell_size_y):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=line_color)

    y = image.height - 1
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill=line_color)

    del draw

    return image


def fill_cell(image, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    margin_x, margin_y = margin * cell_size_x, margin * cell_size_y
    x, y, x_dash, y_dash = row + margin_x, col + margin_y, row + cell_size_x - margin_x, col + cell_size_y - margin_y
    ImageDraw.Draw(image).rectangle([(x, y), (x_dash, y_dash)], fill=fill)


def write_cell_text(image, text, pos, cell_size=None, fill='black', margin=0):
    assert cell_size is not None and 0 <= margin <= 1

    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    margin_x, margin_y = margin * cell_size_x, margin * cell_size_y
    x, y = row + margin_x, col + margin_y
    ImageDraw.Draw(image).text((x, y), text=text, fill=fill)


def draw_cell_outline(image, pos, cell_size=50, fill='black'):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    ImageDraw.Draw(image).rectangle([(row, col), (row + cell_size_x, col + cell_size_y)], outline=fill, width=3)


def draw_circle(image, pos, cell_size=50, fill='black', radius=0.3):
    cell_size_x, cell_size_y = get_cell_sizes(cell_size)
    col, row = pos
    row, col = row * cell_size_x, col * cell_size_y
    gap_x, gap_y = cell_size_x * radius, cell_size_y * radius
    x, y = row + gap_x, col + gap_y
    x_dash, y_dash = row + cell_size_x - gap_x, col + cell_size_y - gap_y
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], outline=fill, fill=fill)


def draw_border(image, border_width=1, fill='black'):
    width, height = image.size
    new_im = Image.new("RGB", size=(width + 2 * border_width, height + 2 * border_width), color=fill)
    new_im.paste(image, (border_width, border_width))
    return new_im


def draw_score_board(image, score, board_height=30):
    im_width, im_height = image.size
    new_im = Image.new("RGB", size=(im_width, im_height + board_height), color='#e1e4e8')
    new_im.paste(image, (0, board_height))

    _text = ', '.join([str(round(x, 2)) for x in score])
    ImageDraw.Draw(new_im).text((10, board_height // 3), text=_text, fill='black')
    return new_im
