import glob
import os
import loguru
import hashlib
import smtplib
import argparse
import subprocess, shlex, pathlib
import datetime
import numpy as np
import librosa

import pandas as pd
from dotenv import dotenv_values

logger = loguru.logger

secrets = dotenv_values(".env")


def get_sample_time(filename, sample_pos, sr=16000, type='dogmic') -> datetime.datetime:
    # start_time = os.path.basename(filename)
    start_time = os.path.basename(filename)
    start_time = '.'.join(start_time.split('.')[:-1])
    if type=='dogmic':
        # the format is YYYYMMDDHHMMSS, so we can convert it to seconds
        start_time = datetime.datetime.strptime(start_time, "%Y%m%d%H%M%S")
    elif type=='camera':
        # the format is 1_YYYY-MM-DD_HH-MM-SS_XXXX, so we can convert it to seconds
        start_time = datetime.datetime.strptime(start_time, "1_%Y-%m-%d_%H-%M-%S_%f")
        start_time = start_time - datetime.timedelta(hours=1)  # the file time is the end of the recording, so we need to subtract 1 hour
    interval_secs = sample_pos / sr
    res = start_time + datetime.timedelta(seconds=int(interval_secs))
    return res


def test_peaks(y, peak_pos, sr, window_duration=0.25,num_show=0):
    '''Validate the detected amplitude peaks to see they are dog barks
    we look at the main frequencies in the stft amplitude spectrum
    
    For our dog, the peaks are:

    Parameters:
    -----------
    y: np.ndarray
        The audio signal.
    peak_pos: np.ndarray
        The positions of the detected peaks in the audio signal.
    sr: int
        The sample rate of the audio signal.
    window_duration: float
        The duration of the window to analyze around each peak in seconds.
    num_show: int
        The number of peaks to show in the plot for debugging.

    Returns:
    --------
    verified_peaks: list
        A list of verified peaks that are likely to be dog barks.
    not_barks: list
        A list of peaks that are not likely to be dog barks.
    '''
    verified_peak_pos = []
    verified_peaks = []
    not_barks = []
    logger.debug(f'validating {len(peak_pos)} peaks')
    last_peak = 0
    num_tested = 0
    for cpeak in peak_pos:
        if cpeak < last_peak + sr * window_duration:
            continue
        last_peak = cpeak
        num_tested += 1
        start_sample = np.max([0,int(cpeak-sr*window_duration)])
        end_sample = np.min([len(y),int(cpeak+sr*window_duration)])
        y_cut = y[start_sample:end_sample]
        res = librosa.stft(y_cut)
        res = np.abs(res)
        # res = librosa.amplitude_to_db(np.abs(res))
        res_mean = res.mean(axis=1)
        # normalize the mean to sum area=1
        res_mean /= np.sum(res_mean)
        # compare the to the expected bark frequencies
        # dist = np.sum(np.abs(res_mean-bark_profile))
        # print(f'peak {cpeak} has distance {dist}')
        int_freq = np.sum(res_mean[20:80])+np.sum(res_mean[95:125])
        bark_amp =  int_freq / (np.sum(res_mean) - int_freq)
        # plt.title(f'Peak at {cpeak}, distance {dist}')
        # if num_tested < num_show:
        #     plt.figure()
        #     plt.imshow(res, aspect='auto', origin='lower')
        #     plt.title(f'Peak at {cpeak}, bark_amp {bark_amp}')
        if bark_amp > 0.5:
            verified_peaks.append(y_cut)
            verified_peak_pos.append(cpeak)
        else:
            not_barks.append(y_cut)
    logger.debug('found %d verified peaks, %d not barks' % (len(verified_peaks), len(not_barks)))
    return verified_peak_pos, verified_peaks, not_barks


def calculate_barks(filename: str, bark_threshold: float = 0.3, bark_max_interval: float = 10, type='camera'):
    # get all the files in the base_dir that match the date
    barks = pd.DataFrame(columns=['start_samples', 'end_samples', 'start_time', 'end_time', 'duration', 'num_barks', 'date'])

    for file in [filename]:
        logger.info('processing file %s' % file)
        y, sr = librosa.load(file)
        start_time = get_sample_time(file, 0, sr=sr, type=type)
        logger.info('start time: %s' % start_time)
        # identify bark events
        peak_pos = np.where(y > bark_threshold)[0]
        # peak_pos = validate_peaks(y, peak_pos, sr)
        peak_pos, ok_barks, not_barks = test_peaks(y, peak_pos, sr)
        if len(peak_pos) == 0:
            logger.info("No barks detected.")
            continue
        peak_index = 0
        done = False
        while not done:
            # logger.info('current peak: %d (position %f)'% ( peak_index, peak_pos[peak_index]))
            next_peak_index = peak_index + 1
            while next_peak_index < len(peak_pos) and peak_pos[next_peak_index] - peak_pos[peak_index] < bark_max_interval * sr:
                next_peak_index += 1
            # logger.info(f'next peak index: {next_peak_index} (position {peak_pos[next_peak_index-1]})')
            # we have a bark event from peak_index to next_peak_index
            start_sample = peak_pos[peak_index]
            end_sample = peak_pos[next_peak_index - 1] + bark_max_interval * sr
            # start_time_event = datetime.timedelta(seconds=start_sample / sr) + start_time
            # end_time_event = datetime.timedelta(seconds=end_sample / sr) + start_time
            start_time_event = get_sample_time(file, sample_pos=start_sample, sr=sr, type=type)
            end_time_event = get_sample_time(file, sample_pos=end_sample, sr=sr, type=type)
            duration = (end_time_event - start_time_event)
            num_barks = next_peak_index - peak_index
            
            # Create a new row as a DataFrame and concatenate it
            new_row = pd.DataFrame({
                'start_samples': [start_sample],
                'end_samples': [end_sample],
                'start_time': [start_time_event],
                'end_time': [end_time_event],
                'duration': [duration],
                'num_barks': [num_barks],
                'date': [start_time_event.date()]
            })
            barks = pd.concat([barks, new_row], ignore_index=True)
            
            peak_index = next_peak_index
            if peak_index >= len(peak_pos):
                done = True
    return barks


def mkv_to_mp3(mkv_path, stream_index=0, out_path=None, vbr_quality=2):
    logger.info(f"Extracting audio from {mkv_path}")
    mkv_path = pathlib.Path(mkv_path)
    if out_path is None:
        out_path = mkv_path.with_suffix(".mp3")
    cmd = f'ffmpeg -hide_banner -y -i "{mkv_path}" -vn -map 0:a:{stream_index} -c:a libmp3lame -q:a {vbr_quality} "{out_path}"'
    subprocess.run(shlex.split(cmd), check=True)
    return str(out_path)


def send_email(recipient, subject, body, user='irrigation.computer.amnon@gmail.com', pwd=None, smtp_server='smtp-relay.sendinblue.com', smtp_port=587, smtp_user='sugaroops@yahoo.com'):
    '''this is for the sendinblue smtp email server
        for gmail use:
        smtp_server='smtp.gmail.com'
    '''
    if pwd is None:
        pwd = secrets.get('EMAIL_PASSWORD', None)
        if pwd is None:
            logger.warning('EMAIL_PASSWORD not in .env - please set')
    if smtp_server is None:
        smtp_server = secrets.get('EMAIL_SMTP_SERVER', None)
        if smtp_server is None:
            logger.warning('EMAIL_SMTP_SERVER not in .env - please set')
            return False
    if smtp_user is None:
        smtp_user = secrets.get('EMAIL_SMTP_USER', None)
        if smtp_user is None:
            logger.warning('EMAIL_SMTP_USER not in .env - please set')
            return False

    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        logger.debug('connecting to email server %s on port %d' % (smtp_server, smtp_port))
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.ehlo()
        server.starttls()
        server.login(smtp_user, pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        logger.info('sent email: subject %s to %s' % (SUBJECT, TO))
        return True
    except Exception as err:
        logger.warning('failed to send email: subject %s to %s. error %s' % (SUBJECT, TO, err))
        return False


def calculate_md5(file):
    '''calculate the md5 checksum for a given file

    Parameters
    ----------
    file : str
        The path to the file for which to calculate the md5 checksum.

    Returns
    -------
    str
        The md5 checksum of the file.
    '''
    with open(file, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    hash = file_hash.hexdigest()
    logger.info(f"MD5 for {file} is {hash}")
    return hash


def find_new_files(dir='/Users/amnon/Downloads/', type='mkv'):
    '''Get list of new video file that still don't have .md5 file (and are not empty)
    
    Parameters
    ----------
    dir : str
        The directory to search for new video files.
    type : str
        The file extension of the video files to search for.
    
    Returns
    -------
    list
        A list of new video files that need to be processed.
    '''
    new_files = []
    files = glob.glob(os.path.join(dir, f'*.{type}'))
    for f in files:
        if not os.path.exists(f + '.md5') and os.path.getsize(f) > 0:
            new_files.append(f)
    if new_files:
        logger.info(f"Found {len(new_files)} files")
    else:
        logger.info("No new files found.")
    return new_files


def pipeline(dir='/Users/amnon/Downloads/'):
    '''Perform the main pipeline processing:
    1. identify new video files (without .md5)
    2. calculate md5 checksums
    3. save checksums to .md5 files
    4. send md5 as email
    5. extract audio
    6. identify barks
    7. save bark to log file
    '''
    new_files = find_new_files(dir)
    mail_lines = []
    if len(new_files) == 0:
        return

    for f in new_files:
        logger.info(f"Processing file: {f}")
        # calculate md5 and save to X.md5
        md5_hash = calculate_md5(f)
        if md5_hash:
            with open(f + '.md5', 'w') as md5_file:
                md5_file.write(md5_hash)
            mail_lines.append(f"MD5 for {f}: {md5_hash}")
        else:
            logger.warning(f"Failed to calculate MD5 for {f}")
        mp3_file = mkv_to_mp3(f)
        if mp3_file:
            logger.info(f"Extracted audio to {mp3_file}")
        else:
            logger.warning(f"Failed to extract audio from {f}")
            continue
        # identify barks
        barks = calculate_barks(mp3_file, bark_threshold=0.3, bark_max_interval=10, type='camera')
        logger.info(f"Identified {len(barks)} bark events in {mp3_file}, total barks duration {barks['duration'].sum()}")
        with open('barks_log.tsv', 'a') as bark_log:
            if barks is not None and len(barks) > 0:
                bark_log.write(barks.to_csv(sep='\t', index=False, header=not os.path.exists('barks_log.tsv')))
        # delete the mp3 file
        os.remove(mp3_file)

    if mail_lines:
        send_email(secrets.get('TARGET_EMAIL'), "MD5 Checksums", "\n".join(mail_lines))

    # Done processing all files
    logger.info("Pipeline processing complete.")


def main():
    parser = argparse.ArgumentParser(description="Process video files for calculating md5 hash, extracting audio and identifying barks")
    parser.add_argument("--dir", type=str, default="/Users/amnon/Downloads/", help="Directory to scan for video files")
    args = parser.parse_args()
    pipeline(dir=args.dir)


if __name__ == "__main__":
    main()

