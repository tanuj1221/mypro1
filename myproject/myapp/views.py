# myapp/views.py


from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import re
from datetime import datetime
from django.shortcuts import render

from django.http import JsonResponse
from .forms import UploadFileForm
import os

from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse

# Your views.py file
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # This is a simplistic check. In a real application, you should use Django's
        # authentication system to handle passwords securely.
        if email == 'super@com' and password == 'drac1221':
            # If the login is successful, redirect to the index page.
            return HttpResponseRedirect(reverse('index'))
        else:
            # If the login fails, return to the login page with an error message.
            context = {'error': 'Invalid credentials'}
            return render(request, 'myapp/main.html', context)

    # If it's a GET request, just display the login page.
    return render(request, 'myapp/main.html')

# Your index view
def index(request):
    context = {}  # Your context dictionary, if needed
    return render(request, 'myapp/index.html', context)

def file_upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print(request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            save_path = './main12.xlsx'  # Use the desired file name
            if os.path.exists(save_path):
                os.remove(save_path)
            with open(save_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            # Redirect to the index view to reload the page
            return redirect('index')  # Make sure 'index' is the name of your URL pattern for the index view
    # If the request is not POST or form is not valid, return an error message
    return JsonResponse({"error": "Failed to upload."}, status=400)


def load_and_prepare_data(file_path):
    data = pd.read_excel(file_path)
    
    # Handling mixed date formats
    data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce', dayfirst=True)
    
    # Optionally, filter out rows with NaT in 'DATE' if necessary
    # data = data.dropna(subset=['DATE'])
    
    return data

def find_time(text):
    # Look for the pattern "At HH:MM" in the text
    time_pattern = r"At (\d{2}:\d{2})"
    matches = re.findall(time_pattern, str(text))
    return matches[0] if matches else None


# Plot the distribution of categorized times
def plot_time_distribution(data):
    # First, extract and categorize times from the 'INCIDENT' column or any relevant column
    data['TIME'] = data['INCIDENT'].apply(find_time)  # Assuming 'INCIDENT' is the column to extract from
    data['TIME_OF_DAY'] = data['TIME'].apply(categorize_time)
    
    # Set the style of the visualization to a dark theme
    sns.set_theme(style="whitegrid")

    # Customize the darkgrid style
    #plt.style.use('dark_background')
    sns.set_context("talk", font_scale=0.8)
    
    # Define a list of colors for each time of day category
    colors = ['#FFD700', '#FFA07A', '#20B2AA', '#778899']  # Example colors for Morning, Afternoon, Evening, Night

    # Then, plot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=data['TIME_OF_DAY'], order=['Morning', 'Afternoon', 'Evening', 'Night'], palette=colors)
    
    # Set title and labels with specific font color for the dark theme
    ax.set_title('Distribution of Incidents by Time of Day', fontsize=16, color='black')
    ax.set_xlabel('Time of Day', fontsize=14, color='black')
    ax.set_ylabel('Number of Incidents', fontsize=14, color='black')
    
    # Change the color of the ticks and labels to white
    ax.tick_params(colors='black', which='both')

    # Remove the left spine for a more minimalistic look
    sns.despine(left=True)

    # Set the facecolor of the figure to the dark theme background color
    plt.gcf().set_facecolor('#fff')

    # Make the plot tight layout for better spacing
    plt.tight_layout()

    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, facecolor='#fff')  # Use dark theme face color for the saved figure
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close()  # Close the plot to free memory
    return graph

# Categorize time into morning, afternoon, evening, night
def categorize_time(time_str):
    if time_str is None:
        return None
    hour = int(time_str.split(':')[0])
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour <= 23:
        return 'Evening'
    else:
        return 'Night'

def generate_heatmap_html(file_path):
    # Load the dataset
    data = load_and_prepare_data(file_path)
    
    # Filter out rows where coordinates are missing
    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Creating a map centered at the average location
    map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=6)
    
    # Adding the HeatMap layer to the map
    HeatMap(data[['Latitude', 'Longitude']].values.tolist()).add_to(folium_map)
    
    # Instead of saving, return the HTML embed code
    return folium_map._repr_html_()

def plot_day_distribution(data):
    """Generate a plot for incidents by day of the week with a modern, sleek, minimalistic dark theme."""
    # Extract day names
    day_names = data['DATE'].dt.day_name()
    # Order for sorting days of the week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Creating a categorical type with the specified order
    day_names = pd.Categorical(day_names, categories=day_order, ordered=True)
    
    # Set the style of the visualization to a dark theme
    sns.set_theme(style="whitegrid")
    
    # Customize the darkgrid style
    #plt.style.use('dark_background')
    sns.set_context("talk", font_scale=0.8)

    # Define a list of colors for each day
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4']

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=day_names, order=day_order, palette=colors)
    ax.set_title("Incidents by Day of the Week", fontsize=16, color='black')
    ax.set_xlabel('Day of the Week', fontsize=14, color='black')
    ax.set_ylabel('Number of Incidents', fontsize=14, color='black')

    # Change the color of the ticks and labels to white
    ax.tick_params(colors='black', which='both')
    plt.xticks(rotation=45)
    plt.yticks()

    # Remove the left spine for a more minimalistic look
    sns.despine(left=True)
    
    # Set the facecolor of the figure to the dark theme background color
    plt.gcf().set_facecolor('#fff')

    # Make the plot tight layout for better spacing
    plt.tight_layout()

    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, facecolor='#fff')  # Use dark theme face color for the saved figure
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close()  # Close the plot to free memory
    return graph

def plot_area_distribution(data):
    """Generate a plot for incidents by area with a modern, sleek, minimalistic dark theme."""
    # Set the style of the visualization to a dark theme
    sns.set_theme(style="whitegrid")
    
    # Customize the darkgrid style
    #plt.style.use('dark_background')
    sns.set_context("talk", font_scale=0.8)

    # Define a list of colors for the top 10 areas
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Create the plot
    plt.figure(figsize=(12, 8))
    area_counts = data['AREA'].value_counts().head(10)  # Top 10 areas
    ax = sns.barplot(x=area_counts.values, y=area_counts.index, palette=colors)

    # Set title and labels with a specific font color for the dark theme
    ax.set_title('Top 10 Areas by Number of Incidents', fontsize=16, color='black')
    ax.set_xlabel('Number of Incidents', fontsize=14, color='black')
    ax.set_ylabel('Area', fontsize=14, color='black')

    # Change the color of the ticks and labels to white
    ax.tick_params(colors='black', which='both')

    # Remove the left spine for a more minimalistic look
    sns.despine(left=True)

    # Set the facecolor of the figure to the dark theme background color
    plt.gcf().set_facecolor('#fff')

    # Make the plot tight layout for better spacing
    plt.tight_layout()

    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, facecolor='#fff')  # Use dark theme face color for the saved figure
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close()  # Close the plot to free memory
    return graph

# Function to categorize time into 3-hour intervals
def categorize_into_intervals(time_str):
    if time_str is None or time_str == "Unknown":
        print(time_str)
        print('unklnown')
        return "00:00-02:59"
    hour = int(time_str.split(':')[0])  # Ensure the string is in a valid time format
    if 0 <= hour < 3:
        return '00:00-02:59'
    elif 3 <= hour < 6:
        return '03:00-05:59'
    elif 6 <= hour < 9:
        return '06:00-08:59'
    elif 9 <= hour < 12:
        return '09:00-11:59'
    elif 12 <= hour < 15:
        return '12:00-14:59'
    elif 15 <= hour < 18:
        return '15:00-17:59'
    elif 18 <= hour < 21:
        return '18:00-20:59'
    else:
        return '21:00-23:59'

def generate_prediction_heatmap(prediction_data):
    # Creating a map centered at the average location
    print(prediction_data)
    

    map_center = [prediction_data['Latitude'].mean(), prediction_data['Longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=6)
    
    # Adding the HeatMap layer to the map
    HeatMap(prediction_data[['Latitude', 'Longitude', 'PREDICTED_INCIDENTS']].values.tolist()).add_to(folium_map)
    
    return folium_map._repr_html_()

def plot_time_intervals_distribution(data):
    data['TIME'] = data['INCIDENT'].apply(find_time)  # Assuming 'INCIDENT' is your text column
    data['TIME_INTERVAL'] = data['TIME'].apply(categorize_into_intervals)
    
    # Set the style of the visualization to a dark theme
    sns.set_theme(style="whitegrid")
    
    # Customize the darkgrid style
    #plt.style.use('dark_background')
    sns.set_context("talk", font_scale=0.8)
    
    # Define a list of colors for the time intervals
    colors = sns.color_palette("Spectral", 8)  # Using a predefined Seaborn palette

    plt.figure(figsize=(12, 8))
    interval_order = ['00:00-02:59', '03:00-05:59', '06:00-08:59', '09:00-11:59',
                      '12:00-14:59', '15:00-17:59', '18:00-20:59', '21:00-23:59']
    ax = sns.countplot(y=data['TIME_INTERVAL'], order=interval_order, palette=colors)
    
    # Set title and labels with specific font color for the dark theme
    ax.set_title('Distribution of Incidents by 3-Hour Intervals', fontsize=16, color='black')
    ax.set_xlabel('Number of Incidents', fontsize=14, color='black')
    ax.set_ylabel('Time Interval', fontsize=14, color='black')
    
    # Change the color of the ticks and labels to white
    ax.tick_params(colors='black', which='both')

    # Remove the left spine for a more minimalistic look
    sns.despine(left=True)

    # Set the facecolor of the figure to the dark theme background color
    plt.gcf().set_facecolor('#fff')

    # Make the plot tight layout for better spacing
    plt.tight_layout()

    # Convert plot to PNG image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, facecolor='#fff')  # Use dark theme face color for the saved figure
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    plt.close()  # Close the plot to free memory
    return graph

from django.shortcuts import render, redirect

from sklearn.linear_model import ElasticNet
def index(request):

    if request.method == "POST":
        # Check if the POST request has a file
        if 'file' in request.FILES:
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_file = request.FILES['file']
                temp_save_path = './temp_uploaded_file.xlsx'  # Temporary save path for the uploaded file
                final_save_path = './main12.xlsx'  # The final desired path for the file

                # Save the uploaded file temporarily
                with open(temp_save_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                # Replace the existing file with the uploaded one
                os.replace(temp_save_path, final_save_path)
            # Proceed with further processing

                # Option 1: Redirect to index to refresh the page with the new data
                return redirect('index')  # Make sure 'index' is the name of your URL pattern for this view
                # Option 2: Return a JSON response and handle the page update via JavaScript
                # return JsonResponse({"message": "File successfully uploaded."})
            else:
                return JsonResponse({"error": "Failed to upload file."}, status=400)
    file_path = "./main12.xlsx"  # Update as necessary
    data = load_and_prepare_data(file_path)
    daily_incident_counts = data.groupby(['DATE', 'AREA']).size().reset_index(name='DAILY_INCIDENT_COUNT')
   
    data = pd.merge(data, daily_incident_counts, on=['DATE', 'AREA'], how='left')
    print(data[['DATE', 'AREA','DAILY_INCIDENT_COUNT']])
    data[['DATE', 'AREA','DAILY_INCIDENT_COUNT']].to_csv('main.csv')
   
    
    # Generate visualizations
    area_distribution_img = plot_area_distribution(data)
    day_distribution_img = plot_day_distribution(data)
    time_distribution_img = plot_time_distribution(data)
    time_intervals_distribution_img = plot_time_intervals_distribution(data)
    original_heatmap_html = generate_heatmap_html(file_path)
    
    label_encoder_area = LabelEncoder()
    label_encoder_time_interval = LabelEncoder()
    data['AREA_ENCODED'] = label_encoder_area.fit_transform(data['AREA'])
    data['TIME_INTERVAL_ENCODED'] = label_encoder_time_interval.fit_transform(data['TIME_INTERVAL'])

    data['DATE'] = data['DATE'].dt.dayofyear
    data = data.fillna(0)
    areas = data['AREA'].unique()

    
    context = {
        'area_distribution_img': day_distribution_img,
        'day_distribution_img': area_distribution_img,
        'time_distribution_img': time_distribution_img,
        'time_intervals_distribution_img': time_intervals_distribution_img,
        'areas': areas,
        'original_heatmap_html': original_heatmap_html,
       
    }
        
    if request.method == "POST":
        try:
            
            # Process form data
            date = request.POST.get("date")
            time_interval = request.POST.get("time_interval")
            
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            day_of_year = date_obj.timetuple().tm_yday
                # Process form data
            date = request.POST.get("date")
            area = request.POST.get("area")
            time_interval = request.POST.get("time_interval")
            
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            day_of_year = date_obj.timetuple().tm_yday

            # Transform area and time_interval using the fitted LabelEncoders
            area_encoded = label_encoder_area.transform([area])[0]
            time_interval_encoded = label_encoder_time_interval.transform([time_interval])[0]
            
            # Prepare the input for prediction
            prediction_input = [[day_of_year, area_encoded, time_interval_encoded]]

            areas = data['AREA'].unique()
            prediction_input = []
            valid_areas = []
            for area in areas:
                area_encoded = label_encoder_area.transform([area])[0] if area in label_encoder_area.classes_ else -1
                time_interval_encoded = label_encoder_time_interval.transform([time_interval])[0] if time_interval in label_encoder_time_interval.classes_ else -1
                
                if area_encoded != -1 or time_interval_encoded != -1:
                    prediction_input.append([day_of_year, area_encoded, time_interval_encoded])
                    valid_areas.append(area)

        
            
            # Train the model
            from sklearn.neighbors import KNeighborsRegressor
            X = data[['DATE', 'AREA_ENCODED', 'TIME_INTERVAL_ENCODED']]
            y = data['DAILY_INCIDENT_COUNT']
            model =KNeighborsRegressor()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            interval_order = ['00:00-02:59', '03:00-05:59', '06:00-08:59', '09:00-11:59',
                        '12:00-14:59', '15:00-17:59', '18:00-20:59', '21:00-23:59']

            # Make predictions for all areas
            predictions = model.predict(prediction_input)
            prediction_result = model.predict(prediction_input)[0]
            context.update({'prediction_result':f"crime prediction on {area} is {prediction_result} during {time_interval} on {date}"})

            
            # Create a DataFrame with predicted incidents for each area
            prediction_data = pd.DataFrame({'AREA': areas, 'PREDICTED_INCIDENTS': predictions})
            
            # Merge with the original data to get the coordinates
            prediction_data = pd.merge(prediction_data, data[['AREA', 'Latitude', 'Longitude']].drop_duplicates(), on='AREA', how='left')
            # Assuming this is how you're constructing prediction_inputs
            prediction_inputs = []
            for area in areas:
                area_encoded = label_encoder_area.transform([area])[0] if area in label_encoder_area.classes_ else -1
                for time_interval in interval_order:
                    time_interval_encoded = label_encoder_time_interval.transform([time_interval])[0] if time_interval in label_encoder_time_interval.classes_ else -1
                    prediction_inputs.append([day_of_year, area_encoded, time_interval_encoded])

            # No need to flatten prediction_inputs - it should already be in the correct shape

            # Predict
            predictions = model.predict(prediction_inputs)  # This should now work without error


        

            # # Flatten the list if it's nested
            # prediction_inputs = [item for sublist in prediction_inputs for item in sublist]

            # # Predict
            # predictions = model.predict(prediction_inputs)

            # Step 2: Organize predictions into a table format
            # Structure the predictions into a dictionary {area: {time_interval: prediction, ...}, ...}
            prediction_table = []

            # Assuming 'date' is already defined in your code
            date_str = date  # Ensure this is in the desired string format, e.g., '2024-03-31'
            # First, create a structure to hold both individual and aggregated predictions
            prediction_details = {area: {'time_slots': {}, 'daily_total': 0} for area in areas}

            # Fill in the details for each time slot and calculate daily totals
            for i, area in enumerate(areas):
                daily_total = 0
                for j, time_interval in enumerate(interval_order):
                    prediction_index = i * len(interval_order) + j
                    prediction = predictions[prediction_index]
                    # Store individual time slot predictions
                    prediction_details[area]['time_slots'][time_interval] = prediction
                    # Aggregate predictions for the daily total
                    daily_total += prediction
                # Save the aggregated daily total
                prediction_details[area]['daily_total'] = daily_total

            # Convert to a list format that's easily usable in the template
            # This includes entries for both specific time slots and daily totals
            prediction_table = []
            for area, details in prediction_details.items():
                for time_slot, prediction in details['time_slots'].items():
                    # Add individual time slot predictions
                    prediction_table.append({
                        'Date': date_str,
                        'Area': area,
                        'Time_Slot': time_slot,
                        'Prediction': prediction
                    })
                # Add the daily total as a separate entry, signified by 'All Day' time slot
                prediction_table.append({
                    'Date': date_str,
                    'Area': area,
                    'Time_Slot': 'All Day',
                    'Prediction': details['daily_total']
                })


            # Generate the heatmap
            heatmap_html = generate_prediction_heatmap(prediction_data)
            
            context['heatmap_html'] = heatmap_html
            context['prediction_table'] = prediction_table
        except ValueError as e:
            # Return an error message to the user
            context['error'] = str(e)
            return render(request, 'myapp/index.html', context)
        except Exception as e:
            # Log the error and return a generic error message
            # You should log e somewhere for debugging purposes
            context['error'] = "An unexpected error occurred"
            return render(request, 'myapp/index.html', context)



    
    return render(request, 'myapp/index.html', context)


