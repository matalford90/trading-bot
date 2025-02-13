from flask import Flask, jsonify, request
import data.data_processor as dp  # Adjust the import based on your project structure

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process_market_data():
    """
    Endpoint to process market data.
    
    Optional query parameter:
      - filepath: path to the CSV file. Defaults to "trading_bot/data/market_data.csv"
    """
    filepath = request.args.get('filepath', "trading_bot/data/market_data.csv")
    try:
        data = dp.load_market_data(filepath)
        processed_data = dp.process_data(data)
        # Convert DataFrame to a JSON-friendly format
        # Using 'records' orientation to get a list of dicts
        result = processed_data.to_dict(orient='records')
        return jsonify({"data": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)