curl --header "Content-Type: application/json" \
     --request POST \
     --data '{"name":"Mahindra Xylo E4 BS IV","year":2010,"selling_price":229999,"km_driven":168000,"fuel":"Diesel","seller_type":"Individual","transmission":"Manual","owner":"First Owner","mileage":"14.0 kmpl","engine":"2498 CC","max_power":"112 bhp","torque":"260 Nm at 1800-2200 rpm","seats":7.0}' \
    http://127.0.0.1:8000/predict_item