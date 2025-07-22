===================================================

# **Autotune Application**

===================================================



\*\*GitHub link if you want to download all the files to the project\*\*
\*\*https://github.com/SteveJaman/AutotuneApp.git\*\*
---



### Application Overview:

The "Live" Autotune application applies an autotune effect to audio files, correcting their pitch to align with a specified musical scale. It features a graphical user interface (GUI) for easy interaction and provides visual feedback through spectrum plots of the original and autotuned audio.



### How to Use the Application:



###### 1\.  Launch the Application:



      \* Locate and run the executable file (e.g., 'AutotuneApp.exe'). A window titled "Simple Autotune GUI with Spectrum" will appear.



###### 2\.  Select an Audio File:



      \* Click the "Select Audio File" button.

      \* A file dialog will open. Navigate to the desired '.wav' audio file and select it.

      \* A message box will confirm the selected file's name.



###### 3\.  Enter the Autotune Key:



      \* In the "Enter autotune key (e.g. C:maj, A:min):" field, type the desired musical key for autotuning.

      \* \*\*Examples of valid formats:\*\*

          \* 'C:maj' (C Major)

          \* 'A:min' (A Minor)

      \* The field defaults to "C:maj".



###### 4\.  Run Autotune:



      \* Once an audio file is selected and a key is entered, click the "Run Autotune" button.

      \* The "Run Autotune" button will be disabled during processing, and a progress bar will update to show the autotuning's progress.



###### 5\.  Processing and Playback:



      \* The application will load the audio, perform pitch detection, apply pitch correction based on the selected key, and re-synthesize the audio.

      \* Upon completion, a success message will appear, indicating the processing time, and the autotuned audio will automatically begin playing.

      \* The application will wait for the playback to finish.



###### 6\.  View Spectrum Analysis:



      \* After the autotuning process and playback are complete, two plots will be displayed in the GUI:

          \* "Original Audio Spectrum": Shows the frequency spectrum of the input audio file.

          \* "Autotuned Audio Spectrum": Displays the frequency spectrum of the processed, autotuned audio, allowing for a visual comparison of the effect.



###### 7\.  Reset for New Operation:



      \* After the process is finished, the "Run Autotune" button will be re-enabled, and the progress bar will reset, allowing you to select a new file or key and run the autotune again.



===============================

### Important Notes:

===============================

  \* This application is designed as an offline processing tool due to the computational demands of high-fidelity audio manipulation. It does not perform real-time autotuning.

  \* The application is a standalone executable and does not require Python to be installed on your system to run.

