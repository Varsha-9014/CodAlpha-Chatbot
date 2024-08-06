import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK and SpaCy
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

faq_data = {
    "hello":"Hello! How can i help you",
    "What are the services you provide":"We provide various services through online",
    "Why my account is not opening": "If there is any issue with your account  details contact our customer care services.",
    "Can you help me":"For sure! Tell me how can i help you",
    "What are your working hours":"Our working hours are from 9 AM  -  8:30 PM",
    "What are your hours of operation?": "Our hours of operation are 9 AM to 5 PM, Monday to Friday.",
    "Where are you located?": "We are located at 123 Main Street, Anytown, USA.",
    "How can I contact customer service?": "You can contact customer service at (555) 555-5555 or email us at support@example.com.",
    "What is your return policy?": "Our return policy allows returns within 30 days of purchase with a receipt.",
    "Do you offer international shipping?": "Yes, we offer international shipping to most countries. Shipping fees apply.",
    "How can I track my order?": "You can track your order using the tracking number provided in your confirmation email.",
    "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.",
    "How do I reset my password?": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions.",
    "Can I cancel or modify my order?": "You can cancel or modify your order within 24 hours of placing it. Contact customer service for assistance.",
    "Do you offer gift cards?": "Yes, we offer gift cards in various denominations. They can be purchased online or in-store.",
    "Are your products environmentally friendly?": "We are committed to sustainability and offer a range of environmentally friendly products.",
    "How can I apply for a job at your company?": "You can view current job openings and apply on our careers page.",
    "What is your privacy policy?": "Our privacy policy can be found on our website. It details how we collect, use, and protect your information.",
    "Do you have a loyalty program?": "Yes, we have a loyalty program that offers exclusive discounts and rewards to members.",
    "Can I return an online purchase in-store?": "Yes, online purchases can be returned in-store with a receipt or order confirmation.",
    "How do I know if an item is in stock?": "Product availability is indicated on the product page. If an item is out of stock, you can sign up for notifications.",
    "What is your policy on price matching?": "We offer price matching on identical items sold by select competitors. Contact customer service for more details.",
    "Do you have a size guide?": "Yes, our size guide is available on the product pages to help you find the right fit.",
    "Can I place an order over the phone?": "Yes, you can place an order over the phone by contacting our customer service team.",
    "How do I update my account information?": "You can update your account information by logging into your account and going to the account settings page.",
    "What should I do if I receive a damaged item?": "If you receive a damaged item, please contact customer service immediately for assistance.",
    "Do you offer repair services?": "We offer repair services for select products. Please contact customer service for more information.",
    "Can I expedite my shipping?": "Yes, we offer expedited shipping options at checkout for an additional fee.",
    "How do I subscribe to your newsletter?": "You can subscribe to our newsletter by entering your email address at the bottom of our homepage.",
    "Do you have a physical store?": "Yes, we have several physical store locations. You can find the nearest one on our store locator page.",
    "What is your warranty policy?": "We offer a one-year warranty on all our products. Details can be found on the warranty page.",
    "How can I leave a product review?": "You can leave a product review by going to the product page and clicking on 'Write a Review'.",
    "Do you offer customizations on your products?": "Yes, we offer customization options on select products. Contact customer service for more details.",
    "How can I contact technical support?": "You can contact technical support by calling (555) 555-5555 or emailing techsupport@example.com.",
    "What is your policy on data security?": "We take data security seriously and use advanced measures to protect your information. More details are in our privacy policy.",
    "Can I schedule an appointment for a consultation?": "Yes, you can schedule an appointment for a consultation by visiting our booking page.",
    "What are your shipping rates?": "Shipping rates vary based on location and shipping method. Rates are calculated at checkout.",
    "How can I apply a discount code to my order?": "You can apply a discount code at checkout in the 'Promo Code' field.",
    "What should I do if I forget my account password?": "If you forget your account password, click on 'Forgot Password' on the login page and follow the instructions.",
    "Do you have a mobile app?": "Yes, our mobile app is available for download on the App Store and Google Play.",
    "What is your response time for customer inquiries?": "Our response time for customer inquiries is typically within 24 hours.",
    "Can I change my delivery address after placing an order?": "You can change your delivery address within 24 hours of placing your order by contacting customer service.",
    "Do you offer a student discount?": "Yes, we offer a student discount. Please provide a valid student ID at checkout.",
    "How do I unsubscribe from your mailing list?": "You can unsubscribe from our mailing list by clicking the 'Unsubscribe' link at the bottom of any email.",
    "Do you offer bulk purchase discounts?": "Yes, we offer bulk purchase discounts for large orders. Contact sales@example.com for more details.",
    "What is your policy on out-of-stock items?": "If an item is out of stock, you can sign up to be notified when it becomes available again.",
    "How do I create an account?": "You can create an account by clicking 'Sign Up' on our website and filling out the registration form.",
    "What is your exchange policy?": "Our exchange policy allows exchanges within 30 days of purchase with a receipt.",
    "Can I track my return status?": "Yes, you can track the status of your return using the tracking number provided when you initiated the return.",
    "How can I provide feedback about my shopping experience?": "You can provide feedback about your shopping experience by filling out the form on our feedback page.",
    "Do you offer gift wrapping services?": "Yes, we offer gift wrapping services for a small fee. You can select this option at checkout.",
    "What is your policy on sale items?": "Sale items are final sale and cannot be returned or exchanged unless they are defective.",
    "Can I order a product that is not listed on your website?": "If you're looking for a product that is not listed on our website, please contact customer service for assistance.",
    "Do you offer installation services?": "We offer installation services for select products. Please contact customer service for more information.",
    "How do I delete my account?": "To delete your account, please contact customer service and request account deletion.",
    "Can I request a catalog?": "Yes, you can request a catalog by filling out the form on our catalog request page.",
    "What is your policy on unauthorized charges?": "If you notice unauthorized charges on your account, please contact customer service immediately.",
    "How do I apply for a wholesale account?": "You can apply for a wholesale account by filling out the application form on our wholesale page.",
    "What should I do if I receive the wrong item?": "If you receive the wrong item, please contact customer service immediately for assistance.",
    "Do you offer financing options?": "Yes, we offer financing options through our partner financing companies. Details are available at checkout.",
    "Can I schedule a product demo?": "Yes, you can schedule a product demo by visiting our demo booking page.",
    "How do I update my payment information?": "You can update your payment information by logging into your account and going to the payment settings page.",
    "What is your policy on counterfeit products?": "We take counterfeit products seriously. If you suspect a product is counterfeit, please contact customer service.",
    "Do you offer any referral programs?": "Yes, we offer a referral program where you can earn rewards for referring new customers.",
    "How do I request a refund?": "You can request a refund by contacting customer service and providing your order details.",
    "What is your policy on pre-order items?": "Pre-order items will be shipped as soon as they become available. You will be notified of the estimated shipping date.",
    "Can I combine multiple discount codes?": "No, only one discount code can be applied per order.",
    "How do I contact the billing department?": "You can contact the billing department by emailing billing@example.com or calling (555) 555-5555.",
    "What is your policy on damaged shipments?": "If your shipment arrives damaged, please contact customer service immediately for assistance.",
    "Do you offer any promotions for first-time customers?": "Yes, we offer a special discount for first-time customers. Use the code WELCOME at checkout.",
    "Can I purchase an extended warranty?": "Yes, you can purchase an extended warranty for select products. Contact customer service for more details.",
    "How do I apply for a vendor partnership?": "You can apply for a vendor partnership by filling out the application form on our vendor page.",
    "What is your policy on lost packages?": "If your package is lost, please contact customer service for assistance with tracking and resolution.",
    "Do you offer any seasonal discounts?": "Yes, we offer seasonal discounts during major holidays and sales events. Sign up for our newsletter to stay informed.",
    "Can I pick up my order in-store?": "Yes, you can choose the in-store pickup option at checkout for no additional charge.",
    "What should I do if I have a problem with my order?": "If you have a problem with your order, please contact customer service for assistance.",
    "Do you offer a military discount?": "Yes, we offer a military discount. Please provide a valid military ID at checkout.",
    "How do I become a brand ambassador?": "You can apply to become a brand ambassador by filling out the application form on our brand ambassador page.",
    "Can I get a price adjustment if an item goes on sale after my purchase?": "Yes, we offer price adjustments within 14 days of purchase if an item goes on sale.",
    "What is your policy on partial shipments?": "If your order is shipped in multiple packages, you will receive tracking information for each shipment.",
    "How do I report a technical issue on the website?": "You can report technical issues by contacting techsupport@example.com with a description of the problem.",
    "What is your policy on duplicate charges?": "If you notice duplicate charges on your account, please contact customer service for assistance.",
    "Can I request a product sample?": "Yes, you can request a product sample by filling out the form on our sample request page.",
    "Do you offer any special services for corporate clients?": "Yes, we offer special services for corporate clients. Please contact sales@example.com for more details.",
    "How do I manage my email preferences?": "You can manage your email preferences by logging into your account and going to the email settings page.",
    "What is your policy on unauthorized sellers?": "Purchases from unauthorized sellers are not covered by our warranty. Please buy from authorized retailers.",
    "Can I get a gift receipt?": "Yes, you can request a gift receipt at checkout or by contacting customer service.",
    "What should I do if I have a complaint?": "If you have a complaint, please contact customer service for resolution.",
    "Do you offer a satisfaction guarantee?": "Yes, we offer a satisfaction guarantee. If you're not happy with your purchase, you can return it within 30 days.",
    "How do I sign up for SMS alerts?": "You can sign up for SMS alerts by entering your mobile number on our SMS signup page.",
    "Do you offer any eco-friendly packaging options?": "Yes, we offer eco-friendly packaging options. You can select this option at checkout.",
    "What is your policy on discontinued items?": "Discontinued items are final sale and cannot be returned or exchanged unless they are defective.",
    "How do I find a product's user manual?": "User manuals can be found on the product page or by contacting customer service.",
    "What is your policy on order minimums?": "We do not have order minimums. You can order as little or as much as you need.",
    "Can I reserve an item for future purchase?": "We do not offer item reservations. Please purchase items when they are in stock.",
    "How do I update my shipping address?": "You can update your shipping address by logging into your account and going to the shipping settings page.",
    "What should I do if my order is delayed?": "If your order is delayed, please contact customer service for an update.",
    "Do you offer any services for special occasions?": "Yes, we offer services for special occasions such as weddings and parties. Contact events@example.com for more details.",
    "How do I request a bulk order quote?": "You can request a bulk order quote by filling out the form on our bulk order page.",
    "What is your policy on re-stocking fees?": "We do not charge re-stocking fees for returns.",
    "How's your day going?": "I'm just a bot, but I'm here to help make your day better! How can I assist you?",
    "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
    "What's the weather like?": "I'm not sure about the weather, but I can help you with any questions about our services!",
    "Do you have any recommendations?": "Sure! What are you looking for? We have a variety of products and services to choose from.",
    "What's your favorite product?": "As an AI, I don't have preferences, but our customers love our [best-selling product].",
    "Can you chat with me?": "Of course! What would you like to talk about or ask?",
    "I'm bored. Any suggestions?": "How about exploring our latest collection or reading our blog for some interesting articles?",
    "What's new with you?": "I'm always learning new things to help you better. What's new with you?",
    "Can you tell me a fun fact?": "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3000 years old and still edible!",
    "Do you have a chatbot?": "Yes, I am the chatbot here to assist you with any questions or information you need!",
    "What's your return policy?": "Our return policy allows returns within 30 days of purchase with a receipt.",
    "Can you tell me more about your services?": "Sure! We offer a variety of services including online shopping, customer support, and more. How can we assist you today?",
    "I'm having a bad day": "I'm sorry to hear that. I'm here to help with any questions or support you might need.",
    "Can you make me laugh?": "Why did the scarecrow win an award? Because he was outstanding in his field!",
    "What's your name?": "I'm your friendly customer service bot! How can I help you today?",
    "Do you like music?": "I don't have ears, but I can help you find information about music products or services!",
    "What do you do for fun?": "I enjoy helping customers like you! How can I assist you today?",
    "Any tips for staying productive?": "Try to take regular breaks, stay organized, and prioritize your tasks. And if you need help with our services, I'm here for you!",
    "What can you do?": "I can assist you with questions about our products and services, help track orders, and much more. How can I help you today?",
    "Do you have any hobbies?": "I love helping customers and learning new things to improve my assistance skills. How can I help you today?",
    "What's your favorite movie?": "As an AI, I don't watch movies, but I can help you find information about our products and services!",
    "How do I get started with your services?": "Just ask me any questions you have, and I'll guide you through our services and products!",
    "what are you doing": "Im an ai assistant i don't have specific task to do, how can i help you"
}
# FAQ Data
# Preprocess and Vectorize FAQ Data
def preprocess(text):
    return ' '.join([token.lemma_ for token in nlp(text.lower()) if not token.is_stop and not token.is_punct])

faq_questions = list(faq_data.keys())
preprocessed_questions = [preprocess(question) for question in faq_questions]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)
# Function to Find the Best Matching FAQ
def find_best_match(query, vectorizer, tfidf_matrix, faq_questions):
    preprocessed_query = preprocess(query)
    query_vector = vectorizer.transform([preprocessed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_index = similarities.argmax()
    return faq_questions[best_match_index], similarities[best_match_index]
def get_response(query):
    best_match_question, similarity = find_best_match(query, vectorizer, tfidf_matrix, faq_questions)
    if similarity > 0.2:  # Threshold for determining a good match
        return faq_data[best_match_question]
    else:
        return "I'm sorry, I don't understand your question. Can you please rephrase?"
# Running in Terminal
if __name__ == "__main__":
    while True:
        user_query = input("Ask a question: ")
        response = get_response(user_query)
        print(f"Response: {response}")
