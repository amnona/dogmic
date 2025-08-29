import glob
import os
import loguru
import hashlib
import smtplib
import argparse
import subprocess, shlex, pathlib
from dotenv import dotenv_values

logger = loguru.logger

secrets = dotenv_values(".env")


def mkv_to_mp3(mkv_path, stream_index=0, out_path=None, vbr_quality=2):
    logger.info(f"Extracting audio from {mkv_path}")
    mkv_path = pathlib.Path(mkv_path)
    if out_path is None:
        out_path = mkv_path.with_suffix(".mp3")
    cmd = f'ffmpeg -y -i "{mkv_path}" -vn -map 0:a:{stream_index} -c:a libmp3lame -q:a {vbr_quality} "{out_path}"'
    subprocess.run(shlex.split(cmd), check=True)
    return out_path


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
        logger.debug('sent email: subject %s to %s' % (SUBJECT, TO))
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

