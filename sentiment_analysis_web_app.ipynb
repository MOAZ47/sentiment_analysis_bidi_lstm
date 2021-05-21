{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from numpy import loadtxt\n",
    "import pickle\n",
    "\n",
    "from flask import Flask,render_template,url_for,request\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load model\n",
    "model = load_model('sentiment_analysis.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# loading\n",
    "with open('tokenizer.pickle', 'rb') as handle:\n",
    "    tokenizer = pickle.load(handle)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_review(model, new_sentences, max_length=100, show_padded_sequence=True ):\n",
    "    # Keep the original sentences so that we can keep using them later\n",
    "      # Create an array to hold the encoded sequences\n",
    "    result_dict = dict()\n",
    "    result_dict['Input'] = new_sentences\n",
    "    if isinstance(new_sentences, str):\n",
    "        new_sentences = [' '.join(new_sentences.split(\" \"))]\n",
    "    trunc_type='post' \n",
    "    padding_type='post'\n",
    "\n",
    "    # text to sequence\n",
    "    new_text_sequences = tokenizer.texts_to_sequences(new_sentences)\n",
    "\n",
    "    # Pad all sequences for the new reviews\n",
    "    new_reviews_padded = pad_sequences(new_text_sequences, maxlen=max_length, \n",
    "                                     padding=padding_type, truncating=trunc_type)             \n",
    "\n",
    "    classes = model.predict(new_reviews_padded)\n",
    "\n",
    "    # The closer the class is to 1, the more positive the review is\n",
    "    for x in range(len(new_sentences)):\n",
    "        \n",
    "        # Print its predicted class\n",
    "        if classes[x] > 0.5:\n",
    "            result_dict['Result'] = \"Positive\"\n",
    "            #print(\"Positive\")\n",
    "        else:\n",
    "            result_dict['Result'] = \"Negative\"\n",
    "            #print(\"Negative\")\n",
    "        result_dict['Probability'] = classes[x]\n",
    "    \n",
    "    #return result_dict['Input'], result_dict['Input'], result_dict['Probability']\n",
    "    return result_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [21/May/2021 17:24:57] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [21/May/2021 17:24:57] \"\u001b[33mGET /static/css/styles.css HTTP/1.1\u001b[0m\" 404 -\n",
      "127.0.0.1 - - [21/May/2021 17:25:02] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [21/May/2021 17:25:02] \"\u001b[33mGET /static/css/styles.css HTTP/1.1\u001b[0m\" 404 -\n"
     ]
    }
   ],
   "source": [
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('home.html')\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "    if request.method == 'POST':\n",
    "        message = request.form['message']\n",
    "        \n",
    "        my_prediction = predict_review(model, message, show_padded_sequence=False)\n",
    "        #seq, pred, prob = predict_review(model, message, show_padded_sequence=False)\n",
    "    return render_template('result.html', result_dict = my_prediction)\n",
    "\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True, use_reloader=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
