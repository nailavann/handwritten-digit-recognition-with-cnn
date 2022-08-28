import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

list = [0,1,2,3,4,5,6,7,8,9]


class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.loaded_model = load_model('mnist_model.h5')

        self.initUI()
    
    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0,0,0,0)

        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(300, 300)
        canvas.fill(QtGui.QColor("black"))
        self.label.setPixmap(canvas)
        self.last_x, self.last_y = None, None

        self.prediction = QtWidgets.QLabel('Prediction: ')
        self.prediction.setFont(QtGui.QFont('Monospace', 20))

        self.button_clear = QtWidgets.QPushButton('CLEAR')
        self.button_clear.clicked.connect(self.clear_canvas)

        self.button_save = QtWidgets.QPushButton('PREDICT')
        self.button_save.clicked.connect(self.predict)

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment = QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_save)

        self.setLayout(self.container)
    
    def clear_canvas(self):
        self.label.pixmap().fill(QtGui.QColor('#000000'))
        self.prediction.clear()
        self.update()

    def predict(self):
        s = self.label.pixmap().toImage().bits().asarray(300 * 300 * 4)
        arr = np.frombuffer(s, dtype=np.uint8).reshape((300, 300, 4))
        arr = np.array(ImageOps.grayscale(Image.fromarray(arr).resize((28,28), Image.ANTIALIAS)))
        arr = (arr/255.0)
        arr = np.expand_dims(arr,axis=0)
        prediction_number = self.loaded_model.predict(arr)[0]
        number=list[prediction_number.argmax()]

        print(number)
        self.prediction.setText('Prediction: '+str(number))
    
    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return 
        painter = QtGui.QPainter(self.label.pixmap())

        p = painter.pen()
        p.setWidth(20)
        self.pen_color = QtGui.QColor('#FFFFFF')
        p.setColor(self.pen_color)
        painter.setPen(p)

        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    mainApp = MainWindow()
    mainApp.setWindowTitle('Prediction')
    mainApp.show()
    sys.exit(app.exec_())