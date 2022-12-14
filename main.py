
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


i = 0
X_clicks = []
y_clicks = []
X_carts = []
y_carts = []
X_orders = []
y_orders = []

# Open the JSON file
with open(r"train_part_aa.json") as f:
    while True:
        # Read lines of the file
        line = f.readline()
        # Parse the JSON data
        try:

            data = json.loads(line)
            events = []
            click_list = []
            cart_list = []
            order_list = []
            for i in range(len(data['events'])):
                if data['events'][i]['type'] == 'clicks':
                    click_list.append(data['events'][i]['aid'])
                if data['events'][i]['type'] == 'carts':
                    cart_list.append(data['events'][i]['aid'])
                else:
                    order_list.append(data['events'][i]['aid'])

            for j in range(len(click_list) - 5):
                row = [a for a in click_list[j:j + 5]]
                X_clicks.append(row)
                y_clicks.append(click_list[j + 5])

            for j in range(len(cart_list) - 5):
                row = [a for a in cart_list[j:j + 5]]
                X_carts.append(row)
                y_carts.append(click_list[j + 5])

            for j in range(len(order_list) - 5):
                row = [a for a in order_list[j:j + 5]]
                X_orders.append(row)
                y_orders.append(click_list[j + 5])


        except json.JSONDecodeError:
            continue




X_clicks = np.array(X_clicks)
y_clicks = np.array(y_clicks)
X_carts = np.array(X_carts)
y_carts = np.array(y_carts)
X_orders = np.array(X_orders)
y_orders = np.array(y_orders)



clf1 = RandomForestClassifier(max_depth=2, random_state=0)
clf1.fit(X_clicks, y_clicks)

clf2 = RandomForestClassifier(max_depth=2, random_state=0)
clf2.fit(X_carts, y_carts)

clf3 = RandomForestClassifier(max_depth=2, random_state=0)
clf3.fit(X_orders, y_orders)

filename1 = 'model_clicks.sav'
filename2 = 'model_carts.sav'
filename3 = 'model_orders.sav'
joblib.dump(clf1, filename1)
joblib.dump(clf1, filename2)
joblib.dump(clf1, filename2)
