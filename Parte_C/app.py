import gradio as gr
import pandas as pd
import pickle
import os


# Definiciones basicas
thres = 0.55

popular_providers = ['yahoo', 'gmail', 'hotmail']

MAX_IP_LENGTH = len("255.255.255.255")

def validate_ip(ip_address):
    try:
        aux_address = str(ip_address)
        if len(aux_address) > MAX_IP_LENGTH:
            return 'long_address'
        else:
            return 'short_address'
    except:
        return 'weird_address'

# Funcion para procesar el email
def process_email(email):
    try:
        aux = email.split('@')[1]
        mail = aux.split('.')[0]
        domain = aux.split('.')[1]
        
        if mail in popular_providers:
            email_provider = mail
        else:
            email_provider = 'other'
            
        return domain, email_provider
    except:
        return 'weird', 'weird'
    

# Define params names
PARAMS_NAME = [
        "orderAmount",
        "orderState",
        "paymentMethodRegistrationFailure",
        "paymentMethodType",
        "paymentMethodProvider",
        "paymentMethodIssuer",
        "transactionAmount",
        "transactionFailed",
        "emailDomain",
        "emailProvider",
        "customerIPAddressSimplified",
        "sameCity"
]

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Parte_A", "modelo_proyecto_final.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "data", "categories_ohe_without_fraudulent.pickle")
BINS_ORDER = os.path.join(BASE_DIR, "data", "saved_bins_order.pickle")
BINS_TRANSACTION = os.path.join(BASE_DIR, "data", "saved_bins_transaction.pickle")

# Cargar modelo
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
# Cargar columnas
with open(COLUMNS_PATH, 'rb') as handle:
        ohe_tr = pickle.load(handle)
# Cargar bins order
with open(BINS_ORDER, 'rb') as handle:
    new_saved_bins_order = pickle.load(handle)
# Cargar bins transaction
with open(BINS_TRANSACTION, 'rb') as handle:
    new_saved_bins_transaction = pickle.load(handle)

def predict(*args):
    try:
        answer_dict = {}

        # Procesar el email 
        email = args[8] 
        email_domain, email_provider = process_email(email)
        
        # Procesar la IP
        ip_address = args[9]
        ip_simplified = validate_ip(ip_address)
        
        # Actualizar los argumentos con los valores procesados del email
        args_list = list(args)
        args_list[8] = email_domain    
        args_list[9] = email_provider 
        args_list[10] = ip_simplified  

        
        answer_dict = {
                "orderAmount": [args[0]],
                "orderState": [args[1]],
                "paymentMethodRegistrationFailure": [args[2]],
                "paymentMethodType": [args[3]],
                "paymentMethodProvider": [args[4]],
                "paymentMethodIssuer": [args[5]],
                "transactionAmount": [args[6]],
                "transactionFailed": [args[7]],
                "emailDomain": [email_domain],
                "emailProvider": [email_provider],
                "customerIPAddressSimplified": [ip_simplified],
                "sameCity": [args[10]]
            }
        
        # Crear un dataframe
        single_instance = pd.DataFrame.from_dict(answer_dict)
        
        
        # Manejo de bins
        single_instance["orderAmount"] = single_instance["orderAmount"].astype(float)
        single_instance["orderAmount"] = pd.cut(single_instance['orderAmount'],
                                        bins=new_saved_bins_order, 
                                        include_lowest=True)
        
        single_instance["transactionAmount"] = single_instance["transactionAmount"].astype(float)
        single_instance["transactionAmount"] = pd.cut(single_instance['transactionAmount'],
                                        bins=new_saved_bins_transaction, 
                                        include_lowest=True)


        # Aplicar one hot encoding a la data entrante
        single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)
        
        probabilities = model.predict_proba(single_instance_ohe)
        probability_positive_class = probabilities[0][1]

        # Actualizar el score en función del threshold definido
        score = 1 if probability_positive_class >= thres else 0
        
        # Formatear la respuesta para que sea más clara
        resultado = "FRAUDE DETECTADO❌" if score == 1 else "TRANSACCIÓN LEGÍTIMA✅"
        confianza = f"{probability_positive_class:.2%}"
        
        return resultado, confianza, score
        
    except Exception as e:
        return f"Error: {str(e)}", "Error en cálculo", "Error"


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Predicción de fraude en transacciones 💳
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## Predecir si una transacción es fraudulenta o no.
                """
            )

            orderAmount = gr.Slider(label="Monto de la orden", minimum=10, maximum=353, step=1, randomize=True)
            orderState = gr.Dropdown(label="Estado de la orden", choices=["fulfilled", "pending", "canceled"], value="fulfilled")
            paymentMethodRegistrationFailure = gr.Checkbox(label="¿Método de pago registrado?", value=True)
            paymentMethodType = gr.Dropdown(label="Tipo de método de pago", choices=["card", "bank_transfer", "mobile_money", "cash", "other"], value="card")
            paymentMethodProvider = gr.Dropdown(label="Proveedor de método de pago", choices=["VISA 16 digit", "Mastercard 16 digit", "American Express 15 digit", "JCB 16 digit", "Diners Club 14 digit", "Discover 16 digit", "Maestro 12 digit", "UnionPay 16 digit", "Mir 16 digit", "RuPay 16 digit", "UPI 16 digit", "PayPal 16 digit", "Apple Pay 16 digit", "Google Pay 16 digit", "Samsung Pay 16 digit", "Paytm 16 digit", "PhonePe 16 digit", "Amazon Pay 16 digit", "Shopify Pay 16 digit", "Stripe 16 digit"], value="VISA 16 digit")
            paymentMethodIssuer = gr.Dropdown(label="Emisor de método de pago", choices=["Citizens First Banks", "Solace Banks", "Bank of America", "JPMorgan Chase", "Wells Fargo", "Bank of New York Mellon", "Citibank", "Capital One", "Goldman Sachs", "Morgan Stanley", "Barclays", "HSBC", "Deutsche Bank", "Commerzbank", "BNP Paribas", "Société Générale", "Crédit Agricole", "Natixis", "Crédit Lyonnais", "Crédit Mutuel", "Crédit Foncier", "Crédit du Nord", "Crédit du Sud"], value="Citizens First Banks")
            transactionAmount = gr.Slider(label="Monto de la transacción", minimum=10, maximum=353, step=1, randomize=True)
            transactionFailed= gr.Checkbox(label="¿Transacción fallida?", value=False)
            # vamos a pedir que escriba su email
            email = gr.Textbox(label="Correo electronico", placeholder="ejemplo@gmail.com", type="email")
            # y su ip
            customerIPAddress = gr.Textbox(label="IP del cliente", placeholder="192.168.1.1", type="text")
            sameCity = gr.Checkbox(label="¿La ciudad de destino es la misma que la ciudad de compra ?", value=True)
                        
        with gr.Column():

            gr.Markdown(
                """
                ## Predicción
                """
            )

            resultado_output = gr.Textbox(label="Resultado", interactive=False)
            probabilidad_output = gr.Textbox(label="Probabilidad de Fraude", interactive=False)
            score_output = gr.Textbox(label="Score", interactive=False)

            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
                predict,
                inputs=[
                    orderAmount,
                    orderState,
                    paymentMethodRegistrationFailure,
                    paymentMethodType,
                    paymentMethodProvider,
                    paymentMethodIssuer,
                    transactionAmount,
                    transactionFailed,
                    email,
                    customerIPAddress,
                    sameCity,
                ],
                outputs=[resultado_output, probabilidad_output, score_output],
                api_name="prediccion"
            )

    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science' 
                target='_blank'>Proyecto demo creado en el bootcamp de EDVAI 🤗
            </a>
        </p>
        """
    )

demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
#demo.launch(share=False)