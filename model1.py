import json

appDict = {
  'name': 'messenger',
  'playstore': True,
  'company': 'Facebook',
  'price': 100
}
app_json = json.dumps(appDict)
print(app_json)