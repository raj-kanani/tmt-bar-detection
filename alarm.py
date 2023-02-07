import plyer
import time


def show_notification(message):
    plyer.notification.notify(
        title="Alarm",
        message=message,
        app_name="Alarm App",
        timeout=10
    )


alarm_time = "15:52"
while True:
    current_time = time.strftime("%H:%M")
    if current_time == alarm_time:
        show_notification("Wake Up! Time to start the day.")
        break
    time.sleep(60)
