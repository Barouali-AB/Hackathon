import pandas as pd
import json
import numpy as np
import joblib

cmp = 0
clicks = []
carts = []
orders = []
with open(r"test.jsonl") as f:
    while cmp < 100:
        # Read lines of the file
        line = f.readline()
        try:
            data = json.loads(line)
            for i in range(len(data['events'])):
                if data['events'][i]['type'] == 'clicks':
                    clicks.append(data['events'][i]['aid'])
                elif data['events'][i]['type'] == 'carts':
                    carts.append(data['events'][i]['aid'])
                else:
                    orders.append(data['events'][i]['aid'])
        except json.JSONDecodeError:
            continue

max_clicks = max(set(clicks), key=clicks.count)
max_carts = max(set(clicks), key=carts.count)
max_orders = max(set(clicks), key=orders.count)

submission = pd.DataFrame(columns=['session_type', 'labels'])

clf1 = joblib.load('clicks_model.sav')
clf2 = joblib.load('carts_model.sav')
clf3 = joblib.load('orders_model.sav')


# Open the JSON file
with open(r"test.jsonl") as f:
    while True:
        # Read lines of the file
        line = f.readline()
        # Parse the JSON data
        data = json.loads(line)
        click_list = []
        cart_list = []
        order_list = []
        for i in range(len(data['events'])):
            if data['events'][i]['type'] == 'clicks':
                click_list.append(data['events'][i]['aid'])
            elif data['events'][i]['type'] == 'carts':
                cart_list.append(data['events'][i]['aid'])
            else:
                order_list.append(data['events'][i]['aid'])

        n1 = len(click_list)
        n2 = len(cart_list)
        n3 = len(order_list)

        if (n1 >= 3):
            X_test = [[a for a in click_list[n - 3:n]]]
            X_test = np.array(X_test)

            pred_item1 = clf1.predict(X_test)[0]
            new_row_1 = {'session_type': str(data['session']) + "_clicks", 'labels': pred_item1}
            submission = submission.append(new_row_1, ignore_index=True)

        else:
            submission = submission.append({'session_type': str(data['session']) + "_clicks", 'labels': max_clicks},
                                           ignore_index=True)

        if (n2 >= 2):
            X_test = [[a for a in cart_list[n - 2:n]]]
            X_test = np.array(X_test)

            pred_item2 = clf2.predict(X_test)[0]
            new_row_2 = {'session_type': str(data['session']) + "_carts", 'labels': pred_item2}
            submission = submission.append(new_row_2, ignore_index=True)

        else:
            submission = submission.append({'session_type': str(data['session']) + "_carts", 'labels': max_carts},
                                           ignore_index=True)

        if (n1 >= 2):
            X_test = [[a for a in order_list[n - 2:n]]]
            X_test = np.array(X_test)

            pred_item3 = clf3.predict(X_test)[0]
            new_row_3 = {'session_type': str(data['session']) + "_orders", 'labels': pred_item3}
            submission = submission.append(new_row_3, ignore_index=True)

        else:
            submission = submission.append({'session_type': str(data['session']) + "_orders", 'labels': max_orders},
                                           ignore_index=True)

submission.to_csv('submission.csv')