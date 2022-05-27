
import pygame
import tkinter
import tkinter.filedialog
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
import time
from joblib import dump as Jdump, load as Jload

pygame.init()
window= pygame.display.set_mode((640,710))

pygame.display.set_caption("pattern recognition of handwritten digits")


#colors
WHITE=(255,255,255)

BLACK=(0,0,0)
GREEN=(0,255,0)
GREY=(128,128,128)
RED=(255,0,0)

#font
font= pygame.font.SysFont('Arial',25,False,False)

class InputTextBox:
    
    def __init__(self,text,posX,posY,length,width):
        self.color=BLACK
        self.text=text
        self.textRender=font.render(self.text,True,WHITE)
        self.x=posX
        self.y=posY
        self.length=length
        self.width=width
        self.active=False
    def draw(self): # draw the text box 
        pygame.draw.rect(window,self.color,(self.x,self.y,self.width,self.length)) 
        window.blit(self.textRender,(self.x+5,self.y+self.length//3))

    def press(self): 
        self.active=True
    def unpress(self):
        self.active=False
    def isPressed(self):    # to check if pressed or not 
        return self.active
    def isInTextBox(self,pos): # to check postion
        i,j=pos
        return i>= self.x and i<= self.x+self.width and j>=self.y and j<=self.y+self.length
    def update(self,text): # this will update the text
        self.text+=text
        self.textRender=font.render(self.text,True,WHITE)
    def BackSpace(self):   # this will delet last later
        self.text= self.text[:-1]
        self.textRender=font.render(self.text,True,WHITE)
    def getText(self):
        if self.text=="": 
            return float(-1)
        else:
            return float(self.text)
class Button:
    
    def __init__(self,text,posX,posY,length,width):
        self.color=GREY
        self.text=font.render(text,True,WHITE)
        self.x=posX
        self.y=posY
        self.length=length
        self.width=width
        

    def draw(self): # draw the button 
        pygame.draw.rect(window,self.color,(self.x,self.y,self.width,self.length)) 
        window.blit(self.text,(self.x+self.width//3.4,self.y+self.length//3))

    def isInButton(self,pos):
        i,j=pos
        return i>= self.x and i<= self.x+self.width and j>=self.y and j<=self.y+self.length

    def pressButton(self): # make the color of the button green to indicate it has been pressed
        self.color=GREEN
    def unpresButton(self): # make the color of the button gray 
        self.color=GREY
    def isPressed(self):
        return self.color==GREEN

class Box:
    def __init__(self,row,column,length,dx,dy):
        self.row=row
        self.column=column
        self.length=length
        self.x=row*self.length+dx
        self.y=column*self.length+dy
        self.color=WHITE
    def press(self):
        self.color=BLACK
    def reset (self):
        self.color=WHITE
    def isPress(self):
        return self.color==BLACK
    def isNotPressed(self):
        return self.color==WHITE
    def draw(self):
        pygame.draw.rect(window,self.color,(self.x,self.y,self.length,self.length))
    
    def getLength(self):
        return self.length


class Grid: #this is a 16 X 16 grid which is made for the user test 
    def __init__(self,posX,posY,length,width):
        self.x=posX
        self.y=posY
        self.grid=[]
        self.length=length
        self.width=width
        self.box_length=length//16
    
    def make_grid(self):    # this will make the grid
        for i in range(16):
            self.grid.append([])
            for j in range(16):
                box=Box(i,j,self.box_length,self.x,self.y)
                self.grid[i].append(box)
    def drawGridLines(self):    # to draw the lines of the grid starting from the specified place for the grid 
        for i in range (16):
            pygame.draw.line(window,GREY,(self.x,i*self.box_length+self.y),(self.length,i*self.box_length+self.y))
        for j in range (16):
            pygame.draw.line(window,GREY,(j*self.box_length+self.x,self.y),(j*self.box_length+self.x,self.width))
    
    def reset(self):    # this will reset every box in the grid to white
        for i in range(16):
            for j in range (16):
                self.grid[i][j].reset()
    def allNotPressed(self):     # this will if there is nothing pressed
        flag=True
        for i in range(16):
            for j in range (16):
                 flag= self.grid[i][j].isNotPressed() and flag
        return flag
    def make_pressed(self,pos): # this will make a specific box pressed
        x,y= pos
        x=x-self.x
        y=y-self.y
        row=x//self.box_length
        col=y//self.box_length
        if x>=0 and y>=0:
            self.grid[row][col].press()
    def make_unpressed(self,pos):
        x,y=pos
        x=x-self.x
        y=y-self.y
        row = x//self.box_length
        col=y//self.box_length
        if x>=0 and y>=0:
            self.grid[row][col].reset()
    
    def getValue(self):
        x=[]
        for i in range(16):
            for j in range(16):
                if self.grid[i][j].isPress():
                    x.append(1)
                else:
                    x.append(0)
        return x
            
    def draw(self): # this will draw the grid
        for colum in self.grid:
            for box in colum:
                box.draw()
        self.drawGridLines()
        pygame.display.update()

class Lable:    # this class was made to make creating lables easier 
    def __init__(self,text,posX,posY,color):
        self.x=posX
        self.y=posY
        self.txt=text
        self.font=pygame.font.SysFont("Arial",20,False,False)
        self.color=color
        self.textRender=self.font.render(text,True,self.color)
    def draw(self):
        window.blit(self.textRender,(self.x,self.y))
    def update(self,text):
        self.textRender=self.font.render(text,True,self.color)
    def changeColor(self,newColor):
        self.color=newColor
    def getValue(self):
        return self.txt    


def prompt_file():
   
    top = tkinter.Tk()
    top.withdraw()  # hide window
    file_name = tkinter.filedialog.askopenfilename(parent=top)
    top.destroy()
    return file_name

def drawPage(pageList):
    window.fill(WHITE)
    for i in pageList:
        i.draw()
    pygame.display.update()

def saveFile():
    top = tkinter.Tk()
    top.withdraw()  # hide window
    fileName =tkinter.filedialog.asksaveasfilename()
    top.destroy()
    return fileName
    
def main():
    flag=True
    writeflag=False
    window.fill(WHITE)
    trainingIsDone=False
    
    #set of buttons
    load=Button("Load",200,95,100,200)      # load button in main page
    training=Button("Train",200,200,100,200)   # train button in main page
    testing=Button("Test",200,305,100,200)      # test button in main page
    WriteCarchter=Button("Write",200,410,100,200)   #write button in main page
    back= Button("Go Back",0,0,70,150)          # back button to return to main page
    load_Xtr=Button("X Training",0,75,70,150)   # load X training in load page
    load_Ytr=Button("Y Training",0,150,70,150)  # load y training in laod page
    load_Xte=Button("X Testing",0,75,70,150)    # load X testing in test page
    load_Yte=Button("Y Testing",0,150,70,150)   # load y testing in test page
    train=Button("Train",200,305,100,200)       # train buuton in train page
    test=Button("Test",0,225,70,150)
    reset =Button("reset",490,0,70,150)         # reset button that will reset the grid in write page 
    compare= Button("compare",250,0,70,150)     # this button will check the value the user has enterd 
    tryAgain= Button("try again",250,410,70,170) # a button to enable the user of retrying to draw the digigt 
    loadModel=Button("load model",0,410,70,175) # a button to load an existing model from your divice
    saveModel=Button("save model",0,140,70,175) # a button to enable the user of saving his model
    #text box
    hiddenNeurons = InputTextBox("",150,225,70,150) 
    learningRate= InputTextBox("",150,300,70,150)

    #lable
    hiddenNeuronsLable = Lable("Hidden Neuron",20,245,BLACK)
    learningRateLable= Lable("Learning Rate",20,320,BLACK)
    fileXtr=Lable("",175,95,BLACK)
    fileYtr=Lable("",175,170,BLACK)
    loadModelLable=Lable("",190,430,BLACK)
    fileXte=Lable("",175,95,BLACK)
    fileYte=Lable("",175,170,BLACK)
    trainingDone=Lable("",200,410,GREEN)
    fileNotFound=Lable("",200,410,RED)
    trainingIsNotDone=Lable("you need to do training first",200,310,RED)
    testFileNotFount= Lable("you need to load files",200,410,RED)
    wrongValues=Lable("none of the values of hidden neurons or learning rate should be equal to zero!!!",40,410,RED)
    valueIs=Lable("",200,350,BLACK)
    trainTime=Lable("",200,450,BLACK)
    DoneSaving=Lable("",190,160,GREEN)
    
    featureY=""
    featureX=""
    featureXte=""
    featureYte=""
    loadModelFile=""
   
    #grid

    grid=Grid(0,70,640,710)
    grid.make_grid()
    
    
    #pages

    mainPage=[]         
    mainPage.append(load)
    mainPage.append(training)
    mainPage.append(testing)
    mainPage.append(WriteCarchter)

    loadPage=[]
    loadPage.append(back)
    loadPage.append(load_Xtr)
    loadPage.append(load_Ytr)
    loadPage.append(hiddenNeurons)
    loadPage.append(hiddenNeuronsLable)
    loadPage.append(learningRate)
    loadPage.append(learningRateLable)
    loadPage.append(fileXtr)
    loadPage.append(fileYtr)
    loadPage.append(loadModel)
    loadPage.append(loadModelLable)
    

    trainingPage=[]
    trainingPage.append(back)
    trainingPage.append(train)
    trainingPage.append(trainingDone)
    trainingPage.append(fileNotFound)
    trainingPage.append(trainTime)
    trainingPage.append(saveModel)
    trainingPage.append(DoneSaving)

    testingPage=[]
    testingPage.append(back)
    testingPage.append(load_Xte)
    testingPage.append(load_Yte)
    testingPage.append(fileXte)
    testingPage.append(fileYte)
    testingPage.append(test)
    


    WriteCarchterPage=[]
    WriteCarchterPage.append(back)
    WriteCarchterPage.append(compare)
    WriteCarchterPage.append(reset)
    WriteCarchterPage.append(grid)

    valuePage=[]
    valuePage.append(valueIs)
    valuePage.append(tryAgain)

    drawPage(mainPage)

    
    while(flag):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:   # to quit the program
                flag=False
            if event.type== pygame.KEYDOWN: # if a key is pressed
                if load.isPressed() and hiddenNeurons.isPressed():  # if you are in the load page and the input textbox hiddenNeurons is pressed
                    if event.key == pygame.K_BACKSPACE:  # backspace key is pressed on the keyboard
                        hiddenNeurons.BackSpace()
                        drawPage(loadPage)
                    else:
                        hiddenNeurons.update(event.unicode) # any other key is pressed on the keyboard
                        drawPage(loadPage)
                        pygame.display.update()
                if load.isPressed() and learningRate.isPressed():
                    if event.key == pygame.K_BACKSPACE: # backspace key is pressed on the keyboard
                        learningRate.BackSpace()
                        drawPage(loadPage)
                    else:
                        learningRate.update(event.unicode) # any other key is pressed on the keyboard
                        drawPage(loadPage)                            
            if pygame.mouse.get_pressed()[0]: # left mouse click is pressed
                pos=pygame.mouse.get_pos()
                if not load.isPressed() and not training.isPressed() and not testing.isPressed() and not WriteCarchter.isPressed() and load.isInButton(pos) : # if you are in the main page and pressed on load button
                    load.pressButton()
                if load.isPressed(): # if you are in the load page
                    drawPage(loadPage)
                    if back.isInButton(pos): # if the back button is pressed go back to the main page
                        load.unpresButton()
                        hiddenNeurons.unpress()
                        drawPage(mainPage)
                    if load_Xtr.isInButton(pos):    # load training file X
                        featureX= prompt_file()
                        fileXtr.update(featureX)
                        drawPage(loadPage)
                    if load_Ytr.isInButton(pos):    # load trainging file Y
                        featureY= prompt_file()
                        fileYtr.update(featureY)
                        drawPage(loadPage)
                    if loadModel.isInButton(pos):   # to load an exisiting model 
                        loadModelFile=prompt_file()
                        loadModelLable.update(loadModelFile)
                        drawPage(loadPage)
                        trainingIsDone=True
                        clf=Jload(loadModelFile)

                    if hiddenNeurons.isInTextBox(pos): # press  hiddenNeurons text box
                        hiddenNeurons.press()
                    if not hiddenNeurons.isInTextBox(pos): # unpress hiddenNeurons text box
                        hiddenNeurons.unpress()
                    if learningRate.isInTextBox(pos): # presss learning rate text box
                        learningRate.press()
                    if not learningRate.isInTextBox(pos): # unprss learning rate text box
                        learningRate.unpress()
                        
                if not training.isPressed() and not WriteCarchter.isPressed() and not testing.isPressed() and not load.isPressed() and training.isInButton(pos): # if you are in the main page and pressed on train button
                    training.pressButton()

                if training.isPressed() :
                    fileNotFound.update("")
                    drawPage(trainingPage)
                    pygame.display.update()
                    

                    if back.isInButton(pos): # if go back button is pressed go back to hte main page
                        training.unpresButton()
                        DoneSaving.update("")
                        drawPage(mainPage)
                    if train.isInButton(pos) and not featureY=="" and not featureX=="" : # when you are in the training page and pressd train and files are loaded
                        DoneSaving.update("")
                        if hiddenNeurons.getText()== -1 and learningRate.getText() ==-1: 
                            clf=MLPClassifier(hidden_layer_sizes=(1,))
                            trX=np.genfromtxt (featureX,dtype=None,delimiter=",")
                            trY=np.genfromtxt (featureY,dtype=None,delimiter=",")
                            start=time.time()
                            clf.fit(trX,trY)
                            stop=time.time()
                            trainTime.update(f"Training time: {stop - start}s")
                            trainingDone.update("Done")
                            trainingIsDone=True
                            

                        elif hiddenNeurons.getText()== -1:
                            clf=MLPClassifier(hidden_layer_sizes=(1,),learning_rate_init=learningRate.getText())
                            trX=np.genfromtxt (featureX,dtype=None,delimiter=",")
                            trY=np.genfromtxt (featureY,dtype=None,delimiter=",")
                            start=time.time()
                            clf.fit(trX,trY)
                            stop=time.time()
                            trainTime.update(f"Training time: {stop - start}s")
                            trainingDone.update("Done")
                            trainingIsDone=True
                            
                        elif  learningRate.getText() ==-1:
                            clf=MLPClassifier(hidden_layer_sizes=(int(hiddenNeurons.getText()),))
                            trX=np.genfromtxt (featureX,dtype=None,delimiter=",")
                            trY=np.genfromtxt (featureY,dtype=None,delimiter=",")
                            start=time.time()
                            clf.fit(trX,trY)
                            stop=time.time()
                            trainTime.update(f"Training time: {stop - start}s")
                            trainingDone.update("Done")
                            trainingIsDone=True
                            

                        elif hiddenNeurons.getText()== 0.0 or learningRate.getText() ==0.0:
                            wrongValues.draw()
                            pygame.display.update() 

                        else:
                            clf=MLPClassifier(hidden_layer_sizes=(int(hiddenNeurons.getText()),),learning_rate_init=learningRate.getText())
                        
                            trX=np.genfromtxt (featureX,dtype=None,delimiter=",")
                            trY=np.genfromtxt (featureY,dtype=None,delimiter=",")
                            start=time.time()
                            clf.fit(trX,trY)
                            stop=time.time()
                            trainTime.update(f"Training time: {stop - start}s")
                            trainingDone.update("Done")
                            trainingIsDone=True
                        drawPage(trainingPage)
                        pygame.display.update()    
                    if train.isInButton(pos) and featureY=="" and featureX=="": # if training files are not loaded
                        
                        fileNotFound.update("you need to load files")
                        drawPage(trainingPage)
                        pygame.display.update()    
                    if saveModel.isInButton(pos) and trainingIsDone:# if save model is prssed and traing is done it is going to let the user decide where to save the model
                        filetoSave=saveFile()
                        Jdump(clf,filetoSave)
                        DoneSaving.update("Done")
                        drawPage(trainingPage)  
                if not testing.isPressed() and not load.isPressed() and not WriteCarchter.isPressed() and not training.isPressed() and testing.isInButton(pos): # if you are in  main page and pressed on test button
                    testing.pressButton()

                if testing.isPressed() : # if you are in testing page
                    drawPage(testingPage)
                    if back.isInButton(pos):
                        testing.unpresButton()
                        drawPage(mainPage)
                    if load_Xte.isInButton(pos): # load testing file X
                        featureXte=prompt_file()
                        fileXte.update(featureXte)
                        drawPage(testingPage)
                    if load_Yte.isInButton(pos): # load testing file Y
                        featureYte=prompt_file()
                        fileYte.update(featureYte)
                        drawPage(testingPage)
                    if test.isInButton(pos) and trainingIsDone and not featureXte=="" and not featureYte== "": # if test button is pressed and files are loaded
                        drawPage(testingPage)
                        teX=np.genfromtxt(featureXte,dtype=None,delimiter=",")
                        teY=np.genfromtxt(featureYte,dtype=None,delimiter=",")
                        pre= clf.predict(teX)
                        confusionMatrix=confusion_matrix(teY,pre)
                        df_cm = pd.DataFrame(confusionMatrix, index= [i for i in "1234567890"],columns=[i for i in "1234567890"])
                        pretty_plot_confusion_matrix(df_cm,cmap="viridis")
                    if test.isInButton(pos) and not trainingIsDone: # if the training is not done the user will see a lable saying that training is not done yet
                        trainingIsNotDone.draw() 
                        pygame.display.update()
                    if test.isInButton(pos) and (featureXte=="" or featureYte== ""): # if at least one of the testing file is not loaded user will see a lable saying that files are not loaded
                        testFileNotFount.draw()
                        pygame.display.update()

                        

                if not testing.isPressed() and not load.isPressed() and not WriteCarchter.isPressed() and not training.isPressed() and WriteCarchter.isInButton(pos): # if you are in the main page and write button is pressed
                    WriteCarchter.pressButton()
                
                if WriteCarchter.isPressed(): # you are in the write page
                    
                    if  not compare.isPressed():
                        window.fill(WHITE)      
                        back.draw()
                        reset.draw()
                        compare.draw()
                        if writeflag:
                            grid.make_pressed(pos)  # to prees on the grid
                        writeflag=True
                        grid.draw()                   
                        pygame.display.update()    # note: need to be drwan step by step due to the condition in the middle

                    if back.isInButton(pos) and not compare.isPressed():    # if go back button is pressed then reset the grid and return to the main page
                        WriteCarchter.unpresButton()
                        writeflag=False
                        grid.reset()
                        drawPage(mainPage)

                    if reset.isInButton(pos) and not compare.isPressed(): # if reset button is preseed then reset the grid
                        grid.reset()
                        writeflag=False
                        drawPage(WriteCarchterPage)

                    if compare.isInButton(pos) and trainingIsDone:
                        compare.pressButton()
                        x=grid.getValue()
                        x=np.reshape(x,(1,256))
                        writenPredect=clf.predict(x)
                        value=str(writenPredect[0]%10)
                        valueIs.update("the value you enterd is "+ value)
                        drawPage(valuePage)
                    if tryAgain.isInButton(pos) and compare.isPressed():
                        compare.unpresButton()
                        drawPage(WriteCarchterPage)

                        
            if pygame.mouse.get_pressed()[2]: # right mouse click is pressed 
                pos=pygame.mouse.get_pos()
                if WriteCarchter.isPressed(): # unpress on the grid
                    grid.make_unpressed(pos)
                    drawPage(WriteCarchterPage)

    pygame.quit()
main()