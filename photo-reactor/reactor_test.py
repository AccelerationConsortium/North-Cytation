from photo_reactor import PhotoReactor
from machine import Pin, SoftI2C, I2C
import time
#Upload this to pico

FAN_INDICES = [0,1]
LED_INDICES = [[1,2],[0,3]]

def run_reactor(rpm_target, duration, light_intensity, reactor_num):
    pr = initialize_photoreactor()
    try:
        fan_index = FAN_INDICES[reactor_num]
        led_index_1 = LED_INDICES[reactor_num][0]
        led_index_2 = LED_INDICES[reactor_num][1]

        duty_cycle = pr.initialize_fan(fan_index,starting_duty_cycle=30) #Initialize fan 0 with duty cycle = 30 
        duty_cycle = pr.set_fan_rpm(fan_index,rpm_target,duty_cycle=duty_cycle) #Move fan to target rpm
        pr.set_brightness(led_index_1, light_intensity)
        pr.set_brightness(led_index_2, light_intensity)
        pr.hold_fan_rpm(fan_index,rpm_target,duration,duty_cycle=duty_cycle) #Hold the fan at target rpm for target time
        pr.turn_off_fan(fan_index)
        pr.turn_off_LED(led_index_1)
        pr.turn_off_LED(led_index_2)
    except Exception as e:
        print("Could not run photoreactor: ", e)
        pr.turn_off_fan(fan_index)
        pr.turn_off_LED(led_index_1)
        pr.turn_off_LED(led_index_2)

def initialize_photoreactor():
    pr = PhotoReactor()
    #Add fans
    pr.add_fan(SoftI2C(scl=Pin(9),sda=Pin(8)))
    pr.add_fan(SoftI2C(scl=Pin(7),sda=Pin(6)))

    #Add leds
    pr.add_led(gpio_pin=0)
    pr.add_led(gpio_pin=1)
    pr.add_led(gpio_pin=2)
    pr.add_led(gpio_pin=3)

    print("Photoreactor initialized")
    return pr

def turn_on_reactor_led(reactor_num, intensity):
    pr = initialize_photoreactor()
    led_index_1 = LED_INDICES[reactor_num][0]
    led_index_2 = LED_INDICES[reactor_num][1]
    pr.set_brightness(led_index_1, intensity)
    pr.set_brightness(led_index_2, intensity)

def turn_off_reactor_led(reactor_num):
    pr = initialize_photoreactor()
    led_index_1 = LED_INDICES[reactor_num][0]
    led_index_2 = LED_INDICES[reactor_num][1]
    pr.turn_off_LED(led_index_1)
    pr.turn_off_LED(led_index_2)

def turn_on_reactor_fan(reactor_num, rpm):
    pr = initialize_photoreactor()
    fan_index = FAN_INDICES[reactor_num]
    duty_cycle = pr.initialize_fan(fan_index,starting_duty_cycle=30) #Initialize fan 0 with duty cycle = 30 
    duty_cycle = pr.set_fan_rpm(fan_index,rpm,duty_cycle=duty_cycle) #Move fan to target rpm

def turn_off_reactor_fan(reactor_num):
    pr = initialize_photoreactor()
    fan_index = FAN_INDICES[reactor_num]
    pr.turn_off_fan(fan_index)
