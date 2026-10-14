"""Recreate the ARIA kappa calibration chart from the project report."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1200
HEIGHT = 820
OUTPUT_PATH = Path("notes") / "aria_spectrum.png"

BACKGROUND = "#FAFBFC"
INK = "#182433"
MUTED = "#596579"
GRID = "#E3E8EF"
BLUE = "#2878B5"
TEAL = "#188977"
ORANGE = "#D97721"
GOLD = "#B58B00"


def dashed_line(draw, start, end, fill, width=3, dash=12, gap=8):
    x0, y0 = start
    x1, y1 = end
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if length == 0:
        return
    dx = (x1 - x0) / length
    dy = (y1 - y0) / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        draw.line(
            (
                (x0 + dx * position, y0 + dy * position),
                (x0 + dx * stop, y0 + dy * stop),
            ),
            fill=fill,
            width=width,
        )
        position += dash + gap


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=36)
    subtitle_font = ImageFont.load_default(size=21)
    label_font = ImageFont.load_default(size=20)
    small_font = ImageFont.load_default(size=17)

    left, top, right, bottom = 150, 135, 1110, 650
    plot_width = right - left
    plot_height = bottom - top

    def point(shared_fraction, kappa):
        return (
            left + shared_fraction * plot_width,
            bottom - kappa * plot_height,
        )

    draw.text((left, 38), "ARIA kappa: shared-ranking calibration", fill=INK, font=title_font)
    draw.text(
        (left, 88),
        "Synthetic calibration points and Grad-Dot value reported in the project notes",
        fill=MUTED,
        font=subtitle_font,
    )

    for tick in range(6):
        fraction = tick / 5
        y = bottom - fraction * plot_height
        x = left + fraction * plot_width
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((left - 56, y - 11), f"{fraction:.1f}", fill=MUTED, font=label_font)
        draw.text((x - 18, bottom + 16), f"{tick * 20}%", fill=MUTED, font=label_font)

    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)

    draw.line((point(0, 0), point(1, 1)), fill=BLUE, width=5)
    calibration = [(0.25, 0.248), (0.50, 0.497), (0.75, 0.748)]
    for shared_fraction, kappa in calibration:
        x, y = point(shared_fraction, kappa)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=TEAL, outline=BACKGROUND, width=2)
        draw.text((x + 13, y - 28), f"{kappa:.3f}", fill=TEAL, font=small_font)

    random_kappa = 0.011
    random_y = point(0, random_kappa)[1]
    dashed_line(draw, (left, random_y), (right, random_y), GOLD, width=3, dash=10, gap=8)
    draw.text((left + 18, random_y - 27), "Random baseline: 0.011", fill=GOLD, font=small_font)

    grad_dot_kappa = 0.129
    grad_dot = point(grad_dot_kappa, grad_dot_kappa)
    dashed_line(draw, (left, grad_dot[1]), grad_dot, ORANGE, width=3, dash=10, gap=8)
    dashed_line(draw, (grad_dot[0], grad_dot[1]), (grad_dot[0], bottom), ORANGE, width=3, dash=10, gap=8)
    x, y = grad_dot
    draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=ORANGE, outline=BACKGROUND, width=2)
    draw.text((x + 18, y - 36), "Grad-Dot: kappa=0.129", fill=ORANGE, font=small_font)
    draw.text((x + 18, y - 14), "about 13% on this calibration", fill=ORANGE, font=small_font)

    x_label = "Shared-ranking component in synthetic scores"
    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((WIDTH - (x_box[2] - x_box[0])) / 2, 705), x_label, fill=INK, font=label_font)

    y_label = "Mean absolute inter-query correlation (kappa)"
    y_image = Image.new("RGBA", (380, 36), (0, 0, 0, 0))
    ImageDraw.Draw(y_image).text((0, 0), y_label, fill=INK, font=label_font)
    y_image = y_image.rotate(90, expand=True)
    image.paste(y_image, (28, top + (plot_height - y_image.height) // 2), y_image)

    legend_y = 758
    draw.line((left, legend_y + 10, left + 45, legend_y + 10), fill=BLUE, width=5)
    draw.text((left + 56, legend_y), "ideal calibration", fill=MUTED, font=small_font)
    draw.ellipse((left + 270, legend_y + 1, left + 288, legend_y + 19), fill=TEAL)
    draw.text((left + 298, legend_y), "reported synthetic points", fill=MUTED, font=small_font)
    draw.ellipse((left + 610, legend_y + 1, left + 628, legend_y + 19), fill=ORANGE)
    draw.text((left + 638, legend_y), "Grad-Dot report", fill=MUTED, font=small_font)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)
    print(f"Saved chart to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
