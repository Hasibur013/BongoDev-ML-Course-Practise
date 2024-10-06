def isLandscape(width, height):
    if width > height:
        return "Landscape"
    else:
        return "Portrait"

width = 500
height = 400
result = isLandscape(width,height)
print(result)