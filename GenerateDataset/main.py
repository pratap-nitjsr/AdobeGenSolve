from Polygon import GeneratePolygonDataset
from Star import GenerateStarDataset
from Circle import GenerateCirclesDataset
from Ellipse import GenerateEllipseDataset
from Line import GenerateLinesDataset
from RoundedRectangle import GenerateRoundedRectangleDataset
from Rectangle import GenerateRectangleDataset

if __name__ == "__main__":
    GenerateRoundedRectangleDataset('train', 1000)
    GenerateRoundedRectangleDataset('test', 200)
    GenerateRoundedRectangleDataset('val', 200)
    print("RoundedRectangle Done..")

    GenerateEllipseDataset('train', 1000)
    GenerateEllipseDataset('test', 200)
    GenerateEllipseDataset('val', 200)
    print("Ellipse Done..")

    GeneratePolygonDataset('train', 1000)
    GeneratePolygonDataset('test', 200)
    GeneratePolygonDataset('val', 200)
    print("Polygon Done..")

    GenerateRectangleDataset('train', 1000)
    GenerateRectangleDataset('test', 200)
    GenerateRectangleDataset('val', 200)
    print("Rectangle Done..")

    GenerateLinesDataset('train', 1000)
    GenerateLinesDataset('test', 200)
    GenerateLinesDataset('val', 200)
    print("Lines Done..")

    GenerateCirclesDataset('train', 1000)
    GenerateCirclesDataset('test', 200)
    GenerateCirclesDataset('val', 200)
    print("Circles Done..")

    GenerateStarDataset('train', 1000)
    GenerateStarDataset('test', 200)
    GenerateStarDataset('val', 200)
    print("Star Done..")
