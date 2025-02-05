# import streamlit as st
# import pandas as pd

# # Title
# st.title("Excel/CSV File Upload and HTML Map Viewer")

# # File Uploader
# uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

# if uploaded_file:
#     # Read file
#     file_extension = uploaded_file.name.split(".")[-1]
#     if file_extension == "csv":
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)
    
#     # Display DataFrame
#     st.write("### Uploaded Data:")
#     st.dataframe(df)

# # Call Button
# if st.button("Call"):
#     st.success("Processing completed! Displaying output files...")

#     # Show output file
#     output_file = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\delivery_trips.xlsx"  # Change this to your actual output file path
#     st.download_button("Download Output File", output_file, file_name="output.csv")

#     # Show HTML file (Map)
#     st.write("### Map Output:")
#     html_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\trips_map.html"  # Change this to your actual HTML file path
#     st.components.v1.html(open(html_file_path, 'r').read(), height=500)












# import streamlit as st
# import pandas as pd

# # Title
# st.title("Excel/CSV File Upload and HTML Map Viewer")

# # File Uploader
# uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

# if uploaded_file:
#     # Read the uploaded file
#     file_extension = uploaded_file.name.split(".")[-1]
#     if file_extension == "csv":
#         df_input = pd.read_csv(uploaded_file)
#     else:
#         df_input = pd.read_excel(uploaded_file)
    
#     # Display uploaded data
#     st.write("### Uploaded Input File:")
#     st.dataframe(df_input)

# # Call Button
# if st.button("Call"):
#     st.success("Processing completed! Displaying output files...")

#     # Simulate an output file (Replace with your real processing logic)
#     # output_file_path = "output.csv"  # Change to your actual output file
#     output_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\delivery_trips.xlsx"  # Change this to your actual output file path

#     df_output = df_input.copy()  # Simulating output (Copying input as output)
#     df_output.to_csv(output_file_path, index=False)

#     # Show output file data in UI
#     st.write("### Processed Output File:")
#     st.dataframe(df_output)

#     # Download button for output file
#     st.download_button("Download Output File", output_file_path, file_name="output.csv")

#     # Show HTML file (Map)
#     st.write("### Map Output:")
#     html_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\trips_map.html"  # Change this to your actual HTML file path

#     st.components.v1.html(open(html_file_path, 'r').read(), height=500)













# import streamlit as st
# import pandas as pd

# # Title
# st.title("Excel/CSV File Upload and HTML Map Viewer")

# # File Uploader
# uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

# if uploaded_file:
#     # Read the uploaded file
#     file_extension = uploaded_file.name.split(".")[-1]
#     if file_extension == "csv":
#         df_input = pd.read_csv(uploaded_file) 
#     else:
        
#         df_input = pd.read_excel(uploaded_file) 
    
#     # Display uploaded data
#     st.write("### Uploaded Input File:")
#     st.dataframe(df_input)

# # Call Button
# if st.button("Call"):
#     st.success("Processing completed! Displaying output files...")

#     # Simulate an output file (Replace with your real processing logic)
#     output_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\delivery_trips.xlsx"
#     # output_file_path = "output.csv"  # Change to your actual output file
#     df_output = df_input.copy()  # Simulating output (Copying input as output)
#     df_output.to_csv(output_file_path, index=False)

#     # Show output file data in UI
#     st.write("### Processed Output File:")
#     st.dataframe(df_output)

#     # Download button for output file
#     st.download_button("Download Output File", output_file_path, file_name="output.csv")

#     # Show HTML file (Map)
#     st.write("### Map Output:")
#     html_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\trips_map.html"
#     # html_file_path = "map.html"  # Change to your actual HTML file
#     st.components.v1.html(open(html_file_path, 'r').read(), height=500)




import streamlit as st
import pandas as pd
import io
from trial2 import main1

# Title
st.title("Excel/CSV File Upload and HTML Map Viewer")

# File Uploader
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the uploaded file
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_excel(uploaded_file, engine="openpyxl")
    
    # Display uploaded data
    st.write("### Uploaded Input File:")
    st.dataframe(df_input)

# Call Button
if st.button("Call"):
    output_df=main1(df_input)
    st.success("Processing completed! Displaying output files...")

    # Simulate an output file (Replace with your real processing logic)
    # output_file_path = "output.csv"  # Change to your actual output file
    # output_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\delivery_trips.xlsx"
    # df_output = df_input.copy()  # Simulating output (Copying input as output)
    # output_df.to_csv(output_df, index=False)    
    output_file_path = "output.csv"  # Specify a valid file path
    output_df.to_csv(output_file_path, index=False)


    # Show output file data in UI
    st.write("### Processed Output File:")
    st.dataframe(output_df)

    # Download button for output file

    # Convert DataFrame to CSV in-memory
    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Move to the start of the buffer


    
    # Show HTML file (Map)
    st.write("### Map Output:")
    # html_file_path = "map.html"  # Change to your actual HTML file
    html_file_path= r"trips_map.html"
    st.components.v1.html(open(html_file_path, 'r').read(), height=500)


    # # Streamlit Download Button
    # st.download_button(
    #     label="Download Output File",
    #     data=csv_buffer,
    #     file_name="output.csv",
    #     mime="text/csv"
    # )


    # st.download_button("Download Output File", output_df, file_name="output.csv")





















# import streamlit as st
# import pandas as pd

# # Title
# st.title("Excel/CSV File Upload and HTML Map Viewer")

# # File Uploader
# uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["csv", "xlsx"])

# if uploaded_file:
#     # Read the uploaded file
#     file_extension = uploaded_file.name.split(".")[-1]
#     if file_extension == "csv":
#         df_input = pd.read_csv(uploaded_file)
#     else:
#         xls = pd.ExcelFile(uploaded_file,engine="calamine")
#         df_input = pd.read_excel(xls)
#         # df_input = pd.read_excel(uploaded_file, engine="openpyxl")  # 🔹 FIXED HERE

    
#     # Display uploaded data
#     st.write("### Uploaded Input File:")
#     st.dataframe(df_input)

# # Call Button
# if st.button("Call"):
#     st.success("Processing completed! Displaying output files...")

#     # Simulate an output file (Replace with your real processing logic)
#     output_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\delivery_trips.xlsx"  # Change to your actual output file
#     df_output = df_input.copy()  # Simulating output (Copying input as output)
#     df_output.to_csv(output_file_path, index=False)

#     # Show output file data in UI
#     st.write("### Processed Output File:")
#     st.dataframe(df_output)

#     # Download button for output file
#     st.download_button("Download Output File", output_file_path, file_name="output.csv")

#     # Show HTML file (Map)
#     st.write("### Map Output:")
#     html_file_path = r"C:\Users\Harsh Bhanushali\OneDrive - Charotar University\Desktop\Nirma Mined E hacthon\UI\trips_map.html"  # Change to your actual HTML file
#     st.components.v1.html(open(html_file_path, 'r').read(), height=500)
