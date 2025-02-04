import requests
import json

url = 'http://127.0.0.1:1234/api/'
params = {"Instrument":"FGBL","Security":"4128839","Date":"20191202","Time":"15:15:04.077796640"}
#params = {"Instrument":"FGBL","Security":"4128839","Date":"20191202","Time":"15:15:04"}


print("Retrieving data...")
response=requests.get(url=url,params=params)
if (response.status_code==200):
	ofile = open('api_data.json', 'w')
	ofile.write(json.dumps(response.json(), indent=1))
	ofile.close()
else:
	print("Connection error!")
	print(str(response.status_code)+" : "+response.reason)
	quit()
print("Done.")
