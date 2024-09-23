import os
import yt_dlp

class FileSizeExceededError(Exception):
    pass

def download_youtube_audio(url, output_path='./', max_filesize=25*1024*1024):  # 25 MB in bytes
    downloaded_filename = None

    def progress_hook(d):
        nonlocal downloaded_filename
        if d['status'] == 'finished':
            downloaded_filename = d['filename']
        elif d['status'] == 'downloading':
            if 'total_bytes' in d and d['total_bytes'] > max_filesize:
                if 'tmpfilename' in d:
                    try:
                        os.remove(d['tmpfilename'])
                        print(f"Deleted temporary file: {d['tmpfilename']}")
                    except OSError as e:
                        print(f"Error deleting temporary file: {e}")
                raise FileSizeExceededError(f"File size exceeds {max_filesize/1024/1024:.2f} MB limit")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path + '%(title)s.%(ext)s',
        'progress_hooks': [progress_hook],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Successfully downloaded audio from: {url}")
        
        # Ensure the filename has .mp3 extension
        if downloaded_filename and not downloaded_filename.endswith('.mp3'):
            new_filename = os.path.splitext(downloaded_filename)[0] + '.mp3'
            if os.path.exists(new_filename):
                downloaded_filename = new_filename

        return downloaded_filename

    except FileSizeExceededError as e:
        print(f"Skipped downloading {url}: {str(e)}")
    except yt_dlp.utils.DownloadError as e:
        print(f"An error occurred while downloading {url}: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {url}: {str(e)}")
    
    return None


def delete_downloaded_file(file_path):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File '{file_path}' successfully deleted.")
        else:
            print(f"Error: File '{file_path}' does not exist.")
    except PermissionError:
        print(f"PermissionError: You do not have permission to delete the file '{file_path}'.")
    except OSError as e:
        print(f"OSError: Failed to delete the file '{file_path}' due to: {e.strerror}.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# Debugging section
if __name__ == "__main__":
    # Example usage for debugging
    debug_urls = [
        "https://www.youtube.com/watch?v=lvAGKsxWAdw",
        "https://www.youtube.com/watch?v=invalid_url",
    ]

    for url in debug_urls:
        download_youtube_audio(url)