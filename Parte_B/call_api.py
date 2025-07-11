import requests

search_api_url = 'http://127.0.0.1:8000/prediction'

# CASO 1
"""
data = {
        "orderAmount" : 18.0,
        "orderState" : "pending",
        "paymentMethodRegistrationFailure" : "True",
        "paymentMethodType" : "card",
        "paymentMethodProvider" : "JCB 16 digit",
        "paymentMethodIssuer" : "Citizens First Banks",
        "transactionAmount" : 18,
        "transactionFailed" : False,
        "emailDomain" : "com",
        "emailProvider" : "yahoo",
        "customerIPAddressSimplified" : "only_letters",
        "sameCity" : "yes"
    }
"""
# CASO 2

data = {
    "orderAmount" : 26.0,
    "orderState" : "fulfilled",
    "paymentMethodRegistrationFailure" : "True",
    "paymentMethodType" : "bitcoin",
    "paymentMethodProvider" : "VISA 16 digit",
    "paymentMethodIssuer" : "Solace Banks",
    "transactionAmount" : 26,
    "transactionFailed" : False,
    "emailDomain" : "com",
    "emailProvider" : "yahoo",
    "customerIPAddressSimplified" : "only_letters",
    "sameCity" : "no"
}



response = requests.post('http://127.0.0.1:8000/prediction', json=data)
print(response.json())
