
--------------------------------------------------------------------------------
-- Literal Homeless History
-- This query creates a households history of literal homelessness up until
-- the current CE enrollment.

-- The subquery starting on line 656 grabs all literal homelessness events for
-- all houshold members and adds the personalID for the head of houshold. 

-- These household literal homeless events are then merged to the HoH's CE
-- Enrollments. The left join that does this starts on line 651 and ends
-- on 724.

-- We finally have all Literal Homeless events of all household members linked
-- to the CE enrollment of the HoH. Now we just need to filter out LH events
-- that occurred after the CE Enrollment, then we aggregate back to the
-- HoH so we can get variables like the total number of LH events and the 
-- average number of LH events per household.

-- In this select statement aggregate to the CE enrollment of the HoH.
SELECT LHCE.CEHouseholdID, COUNT(LHCE.LHPersonalID) TotalLHEvents,
      CASE
        -- Check division by 0 case
         WHEN COUNT(LHCE.LHPersonalID) >0 THEN CAST(COUNT(LHCE.LHPersonalID) AS FLOAT) / CAST(COUNT(DISTINCT(LHCE.LHPersonalID)) AS FLOAT) 
         ELSE 0
       END AvgLHEvents,
       SUM(LHCE.HOH) HOHLHEvents, MAX(LHCE.LHEnrollmentEntryDate) MostRecentLHEnrollment,
       
      MAX( 
            CASE 
              WHEN LHCE.HOH = 1 THEN LHCE.LHEnrollmentEntryDate
              ELSE NULL
            END
      ) AS MostRecentHoHLHEnrollment,
       
       SUM( 
            CASE 
              WHEN LHCE.DaysElapsed < 365 THEN 1
              ELSE 0
            END
       ) AS TotalLHEventsInPast12Months,

       SUM( 
            CASE 
              WHEN LHCE.DaysElapsed < 365 AND LHCE.HOH = 1 THEN 1
              ELSE 0
            END
       ) AS TotalHoHEventsInPast12Months,
       
       SUM( 
            CASE 
              WHEN LHCE.DaysElapsed <= 182 THEN 1
              ELSE 0
            END
       ) AS TotalLHEventsInPast6Months,
       
       SUM( 
            CASE 
              WHEN LHCE.DaysElapsed <= 182 AND LHCE.HOH = 1 THEN 1
              ELSE 0
            END
       ) AS TotalHoHEventsInPast6Months,
       
       CASE -- Need to account for division by zero
         WHEN MIN(LHCE.DaysElapsed) < 365 THEN CAST( SUM( 
                                                    CASE 
                                                      WHEN LHCE.DaysElapsed < 365 THEN 1
                                                      ELSE 0
                                                    END
                                                   ) AS FLOAT) / CAST(COUNT(DISTINCT(CASE 
                                                                                      WHEN LHCE.DaysElapsed < 365 THEN LHCE.LHPersonalID
                                                                                      ELSE NULL
                                                                                      END)) 
                                                                 AS FLOAT)
                                               
         ELSE 0
         END AS AverageLHEventsInPast12Months,
         
            CASE -- Need to account for division by zero
               WHEN MIN(LHCE.DaysElapsed) <= 182 THEN CAST(SUM( 
                                                         CASE 
                                                           WHEN LHCE.DaysElapsed <= 182 THEN 1
                                                           ELSE 0
                                                         END
                                                         ) AS FLOAT) / CAST(COUNT(DISTINCT(CASE 
                                                                                              WHEN LHCE.DaysElapsed < 365 THEN LHCE.LHPersonalID
                                                                                              ELSE NULL
                                                                                              END)) 
                                                                        AS FLOAT)
                                                      
         ELSE 0
         END AS AverageLHEventsInPast6Months
        
     
       
    FROM
    (   
        SELECT CE.PersonalID HoHPersonalID, CE.CEHouseholdID, CE.EnrollmentEntryDate CEEnrollmentEntryDate,
               LH.PersonalID LHPersonalID, LH.EnrollmentID LHEnrollmentID,
               LH.HouseholdID LHHouseholdID, LH.EnrollmentEntryDate LHEnrollmentEntryDate,
               DATEDIFF(day, LH.EnrollmentEntryDate, CE.EnrollmentEntryDate) DaysElapsed,
               -- Identify the Head Of Household from the CE enrollment in the
               -- LH records.
               CASE 
                 WHEN CE.PersonalID = LH.PersonalID THEN 1
                 ELSE 0
               END HoH

        FROM
         
               -- CE Enrollements For the Head of Household Only
               (SELECT e1.PersonalID,  e1.EnrollmentEntryDate, e1.HouseholdID CEHouseholdID
                  FROM e_prod.Enrollment as e1
                  WHERE e1.Perspective = 'HeadOfHousehold'
                    AND e1.ProjectName IN ('Coordinated Entry - Priority Pool',
                                           'Coordinated Entry - Chron Hmls Master List',
                                           'Coordinated Entry - Veterans Master List'
                                          )
                  GROUP BY e1.PersonalID,  e1.EnrollmentEntryDate ,e1.EnrollmentID, e1.HouseholdID 
                  -- These are distinct at this level : Single row per household enrollment
                  ) AS CE
                  
                  LEFT JOIN 
        
        
                           -- Literal Homeless Enrollements for all Houshold Member, but we will flag
                           -- the row containing the Head of Household for each Household ID.
                           (SELECT hohids.HOHID, e.PersonalID, e.EnrollmentID, e.HouseholdID, e.EnrollmentEntryDate
                            FROM e_PROD.Enrollment AS e
                             LEFT JOIN dw.EnrollmentNullMapping AS enm ON e.EnrollmentID = enm.EnrollmentID
                             LEFT JOIN dw.NullReport AS nr ON nr.EntryExitUID = enm.EntryExitUID
                             
                             -- ADD HoH PersonalID to all rows of a LH event
                             LEFT JOIN (SELECT hoh.PersonalID HOHID, hoh.HouseholdID
                                        FROM e_PROD.Enrollment hoh
                                        WHERE hoh.Perspective = 'HeadOfHousehold') AS hohids ON e.HouseholdID = hohids.HouseholdID
                                        
                            WHERE  e.Perspective ='Client' AND
                                  
                                   (e.ProjectTypeName IN ('Street Outreach', 'Emergency Shelter',
                                                        'Safe Haven'))
                                   OR(
                                        e.ProjectTypeName IN ('Transitional Housing',
                                                             'PH - Housing with Services',
                                                             'PH - Permanent Supportive Housing',
                                                             'PH - Housing Only', 'PH - Rapid Re-Housing') 
                                      AND (
                                            (
                                             e.LivingSituationName IN ('Place not meant for habitation',
                                                                       'Emergency shelter,  including hotel or motel paid for with emergency shelter voucher',
                                                                       'Safe Haven')
                                             OR (
                                                 e.LivingSituationName IN ('Foster care home or foster care group home',
                                                                            'Hospital or other residential non-psychiatric medical facility',
                                                                            'Jail,  prison or juvenile detention facility',
                                                                            'Long-term care facility or nursing home',
                                                                            'Psychiatric hospital or other psychiatric facility',
                                                                            'Substance abuse treatment facility or detox center')
                                                 AND
                                                 
                                                 nr.InstitutionStayLessThan90Days ='Yes'
                                                 
                                                 AND
                                                 
                                                 nr.HomelessBeforeInstitutionalStay = 'Yes'
                                                )
                                             OR (
                                                e.LivingSituationName IN ('Residential project or halfway house with no homeless criteria',
                                                                          'Hotel or motel paid for without emergency shelter voucher',
                                                                          'Transitional housing for homeless persons',
                                                                          'Staying or living in a friend''s room, apartment or house',
                                                                          'Staying or living in a family member''s room, apartment or house',
                                                                          'Rental by client, with ongoing housing subsidy',
                                                                          'Rental by client, no ongoing housing subsidy',
                                                                          'Owned by client, with ongoing housing subsidy',
                                                                          'Owned by client, no ongoing housing subsidy',
                                                                          'Host home (non-crisis)',
                                                                          'Client doesn''t know',
                                                                          'Client prefers not to answer',
                                                                          'Data not collected')
                                                AND
                                                nr.InstitutionalStayLessThan7Days = 'Yes'
                                                AND
                                                nr.HomelessBeforeInstitutionalStay = 'Yes'
                                                )
                                            )
                                         
                                           )
                                   )
                            GROUP BY hohids.HOHID, e.PersonalID, e.EnrollmentID, e.HouseholdID, e.EnrollmentEntryDate
                           ) AS LH
        -- Merge LH events to CE enrollments. Note that each row in the LH data has
         -- the personal ID of the HoH attached to it. We also only want LH events
         -- that occured pior to the CE enrollments. 

         ON CE.PersonalID = LH.HOHID 
         WHERE (LH.EnrollmentEntryDate < CE.EnrollmentEntryDate)
            OR LH.EnrollmentEntryDate IS NULL
    ) LHCE
GROUP BY LHCE.CEHouseholdID
GO
