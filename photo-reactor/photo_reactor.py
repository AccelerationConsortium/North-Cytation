from lib import EMC2101
from time import sleep
from machine import Pin, SoftI2C, I2C, PWM
import math


class PhotoReactor:

    LEDS = []
    FANS = []

    def __init__(self):
        pass

    def add_fan(self,fan):
        self.FANS.append ( EMC2101.EMC2101(fan) )

    def add_led(self, gpio_pin):
        led_pwm = PWM(Pin(gpio_pin))
        led_pwm.freq(1000) 
        self.LEDS.append(led_pwm)

    #Turn on an LED with specified index
    def turn_on_LED(self,LED_index):
        self.set_brightness(LED_index, 100)

    #Set the brightness of the specified LED to a specific value (duty cycle should be 0 to 100)
    def set_brightness(self, LED_index, duty_cycle):
        DUTY_CYCLE_MAX = 65535
        if 0 <= duty_cycle <= 100:
            self.LEDS[LED_index].duty_u16(int(DUTY_CYCLE_MAX*duty_cycle/100))
        else:
            print("Duty cycle not between 0 and 100%")

    #Turn off an LED with specified index
    def turn_off_LED(self,LED_index):
        self.set_brightness(LED_index, 0)

    #Turn off a fan with a specified index
    def turn_off_fan(self,fan_index):
        self.FANS[fan_index].set_duty_cycle(0)

    #Set the fan duty cycle to a specific value
    def set_fan_duty_cycle(self, fan_index, target_duty_cycle):
        self.FANS[fan_index].set_duty_cycle(target_duty_cycle)

    #Attempt to initialize fan
    #If the fan does not get above 0 rpm, it will try again with a higher duty cycle
    def initialize_fan(self,fan_index,starting_duty_cycle,wait_time=0.3):
        active_fan = self.FANS[fan_index]
        active_fan.set_duty_cycle(starting_duty_cycle)
        sleep(wait_time)
        active_fan_rpm = active_fan.get_fan_rpm()
        print(f'Initial stirring rpm={active_fan_rpm} with duty cycle={starting_duty_cycle}')
        new_duty_cycle = starting_duty_cycle
        while active_fan_rpm == 0:
            active_fan.set_duty_cycle(0)
            new_duty_cycle = new_duty_cycle + 1
            if new_duty_cycle > 100:  
                print("Cannot initialize fan")
                break
            sleep(wait_time)
            active_fan.set_duty_cycle(new_duty_cycle)
            active_fan_rpm = active_fan.get_fan_rpm()
            print(f'Updated rpm={active_fan_rpm} with duty cycle={starting_duty_cycle}')

        return new_duty_cycle

    #Set the RPM of the specified fan
    #Inputs Target rpm, wait_time (pause between adjustments in seconds), duty cycle (initial duty cycle), rpm tolerance (how close should we be to our target RPM?)
    def set_fan_rpm(self,fan_index,target_rpm,duty_cycle=30,wait_time=0.2,rpm_tolerance=10, max_wait=30):
        print("Target rpm", target_rpm)
        active_fan = self.FANS[fan_index]
        within_target = False
        accumulated_time = 0
        while within_target == False:        
            
            current_rpm = active_fan.get_fan_rpm()
            print(f'RPM={current_rpm} for fan {fan_index} with duty cycle={duty_cycle}')

            if abs(target_rpm-current_rpm)<=rpm_tolerance:
                print(f"Rpm obtained +/- {rpm_tolerance} rpm")
                within_target=True
            elif (-target_rpm+current_rpm)>=rpm_tolerance:
                duty_cycle = duty_cycle - 1
            elif (target_rpm-current_rpm)>=rpm_tolerance:
                duty_cycle = duty_cycle + 1

            #How long have we waited to stabilize?
            accumulated_time += wait_time

            #Stop the loop if it has taken too long
            if accumulated_time >= max_wait:
                print(f"RPM did not stabilize over {max_wait} seconds")
                break

            #Do not set the duty cycle higher than 100 or lower than 0
            if (0 <= duty_cycle <= 100): 
                active_fan.set_duty_cycle(duty_cycle)
            sleep(wait_time)

        return duty_cycle

    #Hold the fan at a specific RPM for a given target_time, adjusting the duty cycle as needed
    def hold_fan_rpm(self,fan_index,target_rpm,target_time,duty_cycle=30,wait_time=0.2,rpm_tolerance=10):
        print(f"Holding target rpm {target_rpm} for {target_time} seconds")
        active_fan = self.FANS[fan_index]
        for i in range (0, int(target_time/wait_time)):
            current_rpm = active_fan.get_fan_rpm()
            print(f'RPM={current_rpm} for fan {fan_index} with duty cycle={duty_cycle}')
            if (-target_rpm+current_rpm)>=rpm_tolerance:
                duty_cycle = duty_cycle - 1
            if (target_rpm-current_rpm)>=rpm_tolerance:
                duty_cycle = duty_cycle + 1
            if (0 <= duty_cycle <= 100): 
                active_fan.set_duty_cycle(duty_cycle)
            sleep(wait_time)
        return duty_cycle



