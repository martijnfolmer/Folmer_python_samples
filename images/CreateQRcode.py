"""
    Use the qrcode library to create a qr code from some string. We create a png

    Requres qrcode library : https://pypi.org/project/qrcode/
    pip install qrcode

    Author :        Martijn Folmer (but heavily based on the samples in https://pypi.org/project/qrcode/)
    Date created :  18-06-2026
"""

import qrcode

# Return the
def create_qr_code(data, version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4):

    # Create class
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_correction,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # generate image
    img = qr.make_image(fill_color="black", back_color="white")

    return img


if __name__=="__main__":

    data = "This is a qr code generated with the CreateQRcode.py script"
    version = 1 # dictates the size of the QR code, between 1 and 40
    error_correction = qrcode.constants.ERROR_CORRECT_L #Error correction _L (7% can be corrected), _M (15% can be corrected), _Q (25% can be corrected), _H = (30% can be corrected)
    box_size = 10 # size of each box inside of the qr code
    border = 4 # number of boxes thick that the code should be (4 is the minimum)

    where_to_store = "C:/Users/martijn.folmer/Folmer_python_samples/readme_img/qr_code_img.png"

    # Create the image and store it in the output
    img = create_qr_code(data, version, error_correction, box_size, border)
    img.save(where_to_store)
