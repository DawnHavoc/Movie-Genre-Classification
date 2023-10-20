from flask import render_template,Flask,request
import requests

import os

import pickle
from datetime import datetime
    


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
# def index():
#     context={}
    
#     if request.method =='POST':
        
#         #--------------------Get the coordinates of the enterd city-------------#
#         if 'weather' in request.form:        
#             city_name = request.form['city']
#             context['city']=city_name

#             #----------get the coordinates--------------#
#             coordinates = get_coordinates(city_name)
#             if coordinates:
#                 latitude, longitude = coordinates
#                 context['lat']=latitude
#                 context['lon']=longitude
#         if 'predict' in request.form:
#             myDict = request.form
#             Month = int(myDict["month"])
#             Year = int(myDict["year"])
#             pred = [Year,Month]
#             res=random_Forest.predict([pred])[0]
#             res=round(res,2)
#             context['data']=result

#     #------------------------get the current location's coordinates-----------------------#
#     else:
#         current_location = get_current_location()
#         if current_location:
#             latitude, longitude, city_name= current_location
#         # else:
#             # print("Location information not available.")
#         context['lat']=latitude
#         context['lon']=longitude
#         context['city']=city_name 
    
#     #-------------------get the weather--------------------------#  


#     # Retrieve the API key from an environment variable
#     api_key = config("OPENWEATHERMAP_API_KEY")

#     if api_key is None:
#         raise Exception("API key not found in environment variables.")  
    
#     url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"      # api url
#     response = requests.get(url)
#     data = response.json()      # get the data about weather

#     kelvinTemp = data['main']['temp']
#     celsiusTemp = kelvinTemp-273.15
#     context['celsi']=round(celsiusTemp)
#     feelsLike=data['main']['feels_like']-273.15
#     context['feelsLike']=round(feelsLike)
#     maxTemp=data['main']['temp_max']-273.15
#     context['maxTemp']=round(maxTemp)
#     minTemp=data['main']['temp_min']-273.15
#     context['minTemp']=round(minTemp)
#     humidity=data["main"]["humidity"]
#     context['humidity']=humidity
#     windSpeed=data["wind"]["speed"]
#     context['windSpeed']=windSpeed
#     direction=data['wind']['deg']
#     context['direction']=direction
#     pressure=data["main"]["pressure"]
#     context['pressure']=pressure
#     visibility=data['visibility']
#     context['visibility']=visibility/1000
#     sunriseTime=datetime.fromtimestamp((int)(data["sys"]["sunrise"])).strftime('%I:%M %p')
#     context['sunrise']=sunriseTime
#     sunsetTime=datetime.fromtimestamp((int)(data["sys"]["sunset"])).strftime('%I:%M %p')
#     context['sunset']=sunsetTime

#     cloudiness = data["clouds"]["all"]
#     context['cloud']=cloudiness
    
#     return render_template('homepage.html', data=context)

@app.route('/subpage1')
def subpage1():
    return render_template('historicaldata.html')

@app.route('/subpage2')
def subpage2():
    return render_template('redalertsmap.html')

@app.route('/subpage3')
def subpage3():
    return render_template('helpdesk.html')

# file=open("D:\Projects\CloudBurst\datasets/model.pkl","rb")
# random_Forest=pickle.load(file)
# file.close()

# @app.route('/handle_button_click', methods=['POST'])
# def handle_button_click():
#     if request.method == 'POST':
#         button_action = request.form['button_action']

#         if button_action == 'clicked':
#             # Call your function here
#             result = home()
#             return f'Button clicked! Result: {result}'

    return 'Invalid request'



#@app.route("/", methods=["GET","POST"])
# def home():
#     if request.method=="POST":
#         myDict = request.form
#         Month = int(myDict["Month"])
#         Year = int(myDict["Year"])
#         pred = [Year,Month]
#         res=random_Forest.predict([pred])[0]
#         res=round(res,2)
#         return render_template('D:\Projects\CloudBurst/templates/homepage.html',Month=Month,Year=Year,res=res)
#     return render_template('D:\Projects\CloudBurst/templates/homepage.html')


@app.route('/process', methods=['POST'])
def process():
    
    input_month = request.form['inputMonth']
    input_year= request.form['inputYear']

    # pred = [input_year,input_month]
    # res=random_Forest.predict([pred])[0]
    # res=round(res,2)
    res="hello"

    
    return render_template('homepage.html', output_data=res)
    
if __name__ == "__main__":
    app.run(debug=True)
    # home()
















